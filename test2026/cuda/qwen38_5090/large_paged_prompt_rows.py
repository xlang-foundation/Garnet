"""Explicit larger prompt pages for an I/O study; no serving factory selects this.

Keeps the existing page materialization, visual replacement and view ownership.
Only the allocation granularity changes. Actual native byte/ownership controls
and model parity are required before using this for serving measurements.
"""
from .paged_prompt_rows import PagedPromptRows


class LargePagedPromptRows(PagedPromptRows):
    MAX_PACKED_PAGE_BYTES=256*1024*1024

    def __init__(self,io,prompt_ids,visual_rows=None,visual_starts=None,*,page_size=2048):
        if type(page_size) is not int or page_size not in (1024,2048):
            raise ValueError('Large prompt study requires1024 or2048 token pages')
        # Reuse all existing metadata validation before any native allocation.
        super().__init__(io,prompt_ids,visual_rows,visual_starts,page_size=128)
        if page_size*self.batch*self.HIDDEN*4>self.MAX_PACKED_PAGE_BYTES:
            raise ValueError('Large prompt page exceeds the native256MiB packed bound')
        self.page_size=page_size

    def stats(self):
        return dict(super().stats(),allocation_policy='explicit_large_prompt_pages',
                    max_native_packed_page_bytes=self.MAX_PACKED_PAGE_BYTES,
                    native_byte_or_model_performance_qualified=False)
