"""Study-only bounded prompt materialization using existing native I/O calls.

Retains CPU token IDs, caller-owned visual tensors and at most two packed pages.
The decoder's existing T32 slices and MTP's shifted single-token lookups work
without retaining full-prompt embeddings, copied rows and stacked rows together.
Native component/model qualification is still required. No serving default uses
this class. The caller must keep visual contents and the IO unchanged until close.
"""
from collections import OrderedDict

class PagedPromptRows:
    HIDDEN=5120
    def __init__(self,io,prompt_ids,visual_rows=None,visual_starts=None,*,page_size=128):
        if type(page_size) is not int or page_size not in (32,64,128):raise ValueError('Prompt page size must be32/64/128')
        if not isinstance(prompt_ids,(list,tuple)) or len(prompt_ids) not in (1,2,4,8):raise ValueError('Prompt batch must be1/2/4/8')
        ids=[]
        for row in prompt_ids:
            if not isinstance(row,(list,tuple)) or not 1<=len(row)<=4096 or any(type(t) is not int or not 0<=t<248320 for t in row):
                raise ValueError('Prompt IDs require nonempty bounded rows of exact integer IDs')
            ids.append(tuple(row))
        batch=len(ids)
        if visual_rows is None:visual_rows=[None]*batch
        if visual_starts is None:visual_starts=[-1]*batch
        if not isinstance(visual_rows,(list,tuple)) or not isinstance(visual_starts,(list,tuple)) or len(visual_rows)!=batch or len(visual_starts)!=batch:
            raise ValueError('Prompt visual metadata must match physical batch')
        for row,(visual,start) in enumerate(zip(visual_rows,visual_starts)):
            if type(start) is not int or (visual is None and start!=-1) or (visual is not None and not 0<=start<=len(ids[row])-64):
                raise ValueError('Prompt visual interval must contain exactly64tokens or be absent')
        self.io=io;self.ids=tuple(ids);self.visuals=tuple(visual_rows);self.starts=tuple(visual_starts)
        self.batch=batch;self.lengths=tuple(map(len,ids));self.count=max(self.lengths);self.page_size=page_size
        self.pages=OrderedDict();self.pack_calls=0;self.peak_cached_logical_bytes=0;self.closed=False

    def _open(self):
        if self.closed:raise ValueError('Prompt source is closed')

    def __len__(self):self._open();return self.count

    def _page(self,number):
        if number in self.pages:
            self.pages.move_to_end(number);return self.pages[number]
        # Evict before constructing another page; outstanding caller views retain
        # their native storage/fences and are outside this cache's logical bound.
        if len(self.pages)==2:self.pages.popitem(last=False)
        start=number*self.page_size;end=min(self.count,start+self.page_size);count=end-start
        embedding=[];visuals=[];visual_starts=[]
        for row,ids in enumerate(self.ids):
            tokens=list(ids[start:end]);tokens.extend([0]*(count-len(tokens)))
            embedding.append(self.io.embedding(tokens))
            visual=self.visuals[row];visual_start=self.starts[row]
            first=max(start,visual_start);last=min(end,visual_start+64)
            if visual is None or first>=last:
                visuals.append(None);visual_starts.append(-1)
            else:
                flat=self.io.storage_view(visual,[1,64*self.HIDDEN])
                part=self.io.slice(flat,(first-visual_start)*self.HIDDEN,[1,(last-first)*self.HIDDEN])
                visuals.append(self.io.storage_view(part,[1,last-first,self.HIDDEN]));visual_starts.append(first-start)
        packed=list(self.io.pack_prompt(embedding,visuals,visual_starts))
        if len(packed)!=count:raise ValueError('Native packed page length mismatch')
        self.pages[number]=packed;self.pack_calls+=1
        self.peak_cached_logical_bytes=max(self.peak_cached_logical_bytes,self._cached_bytes())
        return packed

    def __getitem__(self,index):
        self._open()
        if isinstance(index,slice):
            if index.step not in (None,1) or type(index.step) is bool:raise ValueError('Prompt slices require unit stride')
            start,end,step=index.indices(self.count)
            if end-start>self.page_size:raise ValueError('Prompt slice exceeds one decoder page')
            return [self[i] for i in range(start,end,step)]
        if type(index) is not int:raise TypeError('Prompt index must be an integer')
        if index<0:index+=self.count
        if not 0<=index<self.count:raise IndexError('Prompt token index outside source')
        return self._page(index//self.page_size)[index%self.page_size]

    def _cached_bytes(self):return sum(len(page)*self.batch*self.HIDDEN*4 for page in self.pages.values())

    def stats(self):
        self._open()
        return {'physical_batch':self.batch,'tokens':self.count,'lengths':list(self.lengths),'page_size':self.page_size,'cached_pages':list(self.pages),
                'pack_calls':self.pack_calls,'cached_packed_logical_bytes':self._cached_bytes(),'peak_cached_packed_logical_bytes':self.peak_cached_logical_bytes,
                'scope':'Logical packed payload in this two-page cache only; excludes visuals, transient embedding/slice allocations, native allocator/fences and outstanding caller views. Not measured VRAM or a runtime qualification.'}

    def close(self):
        self.pages.clear();self.ids=();self.visuals=();self.starts=();self.io=None;self.closed=True
