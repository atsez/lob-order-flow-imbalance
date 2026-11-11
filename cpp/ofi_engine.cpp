// Streaming order-flow-imbalance engine for LOBSTER level-1 data.
//
// Reads the compact LOBSTER CSV on stdin (header + rows of
// time,type,order_id,size,price,direction,ask1,asksize1,bid1,bidsize1),
// computes the Cont-Kukanov-Stoikov event OFI e_n in one pass, and writes
//
//   ofi_engine events            ->  one e_n per input row
//   ofi_engine grid S E STEP     ->  level-1 state and cumulative OFI as of
//                                    each grid time t = S + k*STEP < E,
//                                    using only events with time <= t
//
// It is the fast path of the study; the Python implementation in
// src/lobofi/features.py is the reference, and tests/test_cpp_engine.py
// checks the two agree exactly on every row of every stock.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct L1 { int64_t pb, qb, pa, qa; };

static inline int64_t ofi(const L1& p, const L1& c) {
    int64_t e = 0;
    if (c.pb >= p.pb) e += c.qb;
    if (c.pb <= p.pb) e -= p.qb;
    if (c.pa <= p.pa) e -= c.qa;
    if (c.pa >= p.pa) e += p.qa;
    return e;
}

// Parse one CSV line. Fields: 0 time, 6 ask1, 7 asksize1, 8 bid1, 9 bidsize1.
static bool parse(char* line, double& t, L1& s) {
    char* f[10];
    int n = 0;
    f[n++] = line;
    for (char* c = line; *c && n < 10; ++c)
        if (*c == ',') { *c = '\0'; f[n++] = c + 1; }
    if (n < 10) return false;
    t = std::strtod(f[0], nullptr);
    s.pa = std::strtoll(f[6], nullptr, 10);
    s.qa = std::strtoll(f[7], nullptr, 10);
    s.pb = std::strtoll(f[8], nullptr, 10);
    s.qb = std::strtoll(f[9], nullptr, 10);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ofi_engine events | grid START END STEP\n");
        return 2;
    }
    const bool grid_mode = std::strcmp(argv[1], "grid") == 0;
    double start = 0, end = 0, step = 1;
    if (grid_mode) {
        if (argc != 5) { std::fprintf(stderr, "grid needs START END STEP\n"); return 2; }
        start = std::atof(argv[2]); end = std::atof(argv[3]); step = std::atof(argv[4]);
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> times;
    std::vector<L1> book;
    std::vector<int64_t> cum;
    times.reserve(1 << 20); book.reserve(1 << 20); cum.reserve(1 << 20);

    char buf[512];
    if (!std::fgets(buf, sizeof buf, stdin)) return 1;  // header
    int64_t running = 0;
    while (std::fgets(buf, sizeof buf, stdin)) {
        double t; L1 s;
        if (!parse(buf, t, s)) continue;
        if (!book.empty()) running += ofi(book.back(), s);
        times.push_back(t); book.push_back(s); cum.push_back(running);
    }
    auto t1 = std::chrono::steady_clock::now();

    if (!grid_mode) {
        int64_t prev = 0;
        for (size_t i = 0; i < cum.size(); ++i) {
            std::printf("%lld\n", static_cast<long long>(cum[i] - prev));
            prev = cum[i];
        }
    } else {
        const long n = std::lround((end - start) / step);
        size_t j = 0;  // events are time-sorted, so one forward pointer suffices
        std::printf("t,idx,bid,bidsize,ask,asksize,cumofi\n");
        for (long k = 0; k < n; ++k) {
            const double t = start + step * static_cast<double>(k);
            while (j < times.size() && times[j] <= t) ++j;
            if (j == 0) { std::fprintf(stderr, "grid starts before first event\n"); return 1; }
            const size_t i = j - 1;  // last event with time <= t
            const L1& s = book[i];
            std::printf("%.9f,%zu,%lld,%lld,%lld,%lld,%lld\n", t, i,
                        (long long)s.pb, (long long)s.qb, (long long)s.pa, (long long)s.qa,
                        (long long)cum[i]);
        }
    }
    auto t2 = std::chrono::steady_clock::now();
    const double parse_s = std::chrono::duration<double>(t1 - t0).count();
    const double total_s = std::chrono::duration<double>(t2 - t0).count();
    std::fprintf(stderr, "events=%zu parse_and_ofi_s=%.4f total_s=%.4f events_per_s=%.0f\n",
                 times.size(), parse_s, total_s, times.size() / parse_s);
    return 0;
}
