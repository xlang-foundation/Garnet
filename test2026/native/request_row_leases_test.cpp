#include "../../src/runtime/scheduler/request_row_leases.h"
#include <iostream>
#include <stdexcept>
using namespace Garnet;
void Check(bool value){if(!value)throw std::runtime_error("Native row lease invariant failed");}
template<class F> void Reject(F fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}Check(rejected);}
int main() {
    RequestRowLeases rows(4);
    Reject([&]{rows.Claim(0);});
    auto a=rows.Claim(1),b=rows.Claim(2),c=rows.Claim(3),d=rows.Claim(4);
    Check(a.row==0&&b.row==1&&c.row==2&&d.row==3);
    Reject([&]{rows.Claim(5);});Reject([&]{rows.Claim(2);});
    Check(!rows.Release(0,b.epoch)&&rows.Matches(0,a.epoch));
    Check(rows.Release(a.row,a.epoch));auto e=rows.Claim(5);
    Check(e.row==a.row&&e.epoch!=a.epoch&&!rows.Release(a.row,a.epoch));
    Check(rows.Matches(b.row,b.epoch)&&rows.Matches(c.row,c.epoch)&&rows.Matches(d.row,d.epoch));
    rows.Clear();Check(!rows.Owned(0)&&!rows.Release(e.row,e.epoch));
    auto again=rows.Claim(1);Check(again.epoch!=a.epoch&&again.epoch!=e.epoch);
    RequestRowLeases rebuilt(4);auto replacement=rebuilt.Claim(1);
    Check(replacement.epoch!=again.epoch&&!rebuilt.Release(replacement.row,again.epoch));
    std::cout<<"native_request_row_leases PASS stable rows, capacity, duplicates, stale release, clear and rebuild\n";
}
