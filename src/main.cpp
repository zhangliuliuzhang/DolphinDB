// Label Propagation Challenge — synchronous LPA on an undirected graph.
// Dependencies: none beyond the C++ standard library and OpenMP.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#else
#include <chrono>
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif

struct Span {
    const char* p;
    uint32_t len;
};

static inline uint64_t hash_str(const char* p, uint32_t len) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)len * 0xff51afd7ed558ccdULL);
    while (len >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        h ^= v;
        h *= 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
        p += 8;
        len -= 8;
    }
    if (len) {
        uint64_t v = 0;
        memcpy(&v, p, len);
        h ^= v;
        h *= 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
    }
    h ^= h >> 32;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 29;
    return h;
}

static inline uint64_t num_hash(uint64_t val, uint32_t len) {
    uint64_t h = val * 0x9E3779B97F4A7C15ULL + len * 0xff51afd7ed558ccdULL;
    h ^= h >> 29;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}

// Parse a pure-digit span (<= 18 digits) into a uint64.
static inline bool parse_num(const char* p, uint32_t len, uint64_t& v) {
    if (len == 0 || len > 18) return false;
    uint64_t x = 0;
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t c = (uint32_t)(unsigned char)p[i] - '0';
        if (c > 9) return false;
        x = x * 10 + c;
    }
    v = x;
    return true;
}

// String interner (non-numeric ids); keys are spans into the input buffer.
struct Interner {
    struct Ent {
        const char* p;
        uint32_t len;
        int32_t id;
        uint64_t h;
        bool used;
    };
    std::vector<Ent> tab;
    int32_t cnt = 0;
    int32_t occ = 0;

    void init(size_t hint) {
        size_t c = 1024;
        while (c < hint * 2) c <<= 1;
        tab.assign(c, {});
    }
    void grow() {
        std::vector<Ent> old;
        old.swap(tab);
        size_t nc = old.size() * 2;
        tab.assign(nc, {});
        for (const Ent& e : old) {
            if (!e.used) continue;
            size_t i = e.h & (nc - 1);
            while (tab[i].used) i = (i + 1) & (nc - 1);
            tab[i] = e;
        }
    }
    size_t mask() const { return tab.size() - 1; }

    int32_t intern(const char* p, uint32_t len, uint64_t h) {
        size_t m = mask(), i = h & m;
        for (;;) {
            Ent& e = tab[i];
            if (!e.used) {
                int32_t id = cnt++;
                e = {p, len, id, h, true};
                if (++occ * 4 > (int32_t)tab.size() * 3) grow();
                return id;
            }
            if (e.h == h && e.len == len && memcmp(e.p, p, len) == 0) return e.id;
            i = (i + 1) & m;
        }
    }
    int32_t insertVal(const char* p, uint32_t len, uint64_t h, int32_t val) {
        size_t m = mask(), i = h & m;
        for (;;) {
            Ent& e = tab[i];
            if (!e.used) {
                e = {p, len, val, h, true};
                if (++occ * 4 > (int32_t)tab.size() * 3) grow();
                return val;
            }
            if (e.h == h && e.len == len && memcmp(e.p, p, len) == 0) return e.id;
            i = (i + 1) & m;
        }
    }
    int32_t lookupOrAssign(const char* p, uint32_t len, uint64_t h,
                           std::atomic<int32_t>& counter) {
        size_t m = mask(), i = h & m;
        for (;;) {
            Ent& e = tab[i];
            if (!e.used) {
                int32_t id = counter.fetch_add(1, std::memory_order_relaxed);
                e = {p, len, id, h, true};
                if (++occ * 4 > (int32_t)tab.size() * 3) grow();
                return id;
            }
            if (e.h == h && e.len == len && memcmp(e.p, p, len) == 0) return e.id;
            i = (i + 1) & m;
        }
    }
};

// Numeric-id interner: keys are (value, length) pairs — injective, no collisions.
struct NumInterner {
    struct Slot {
        uint64_t val;
        uint32_t len;
        int32_t id;
    };
    std::vector<Slot> tab;
    std::vector<std::pair<uint64_t, uint32_t>> idkey;  // per id: (val, len)
    int32_t occ = 0;

    void init(size_t hint) {
        size_t c = 1024;
        while (c < hint * 2) c <<= 1;
        tab.assign(c, {0, 0, -1});
        idkey.reserve(hint + 8);
    }
    void grow() {
        std::vector<Slot> old;
        old.swap(tab);
        size_t nc = old.size() * 2;
        tab.assign(nc, {0, 0, -1});
        for (const Slot& s : old) {
            if (s.id < 0) continue;
            size_t i = num_hash(s.val, s.len) & (nc - 1);
            while (tab[i].id >= 0) i = (i + 1) & (nc - 1);
            tab[i] = s;
        }
    }
    size_t mask() const { return tab.size() - 1; }

    int32_t intern(uint64_t val, uint32_t len) {
        size_t m = mask(), i = num_hash(val, len) & m;
        for (;;) {
            Slot& s = tab[i];
            if (s.id < 0) {
                int32_t id = (int32_t)idkey.size();
                idkey.push_back({val, len});
                s = {val, len, id};
                if (++occ * 4 > (int32_t)tab.size() * 3) grow();
                return id;
            }
            if (s.val == val && s.len == len) return s.id;
            i = (i + 1) & m;
        }
    }
    int32_t insertVal(uint64_t val, uint32_t len, int32_t rowIdx) {
        size_t m = mask(), i = num_hash(val, len) & m;
        for (;;) {
            Slot& s = tab[i];
            if (s.id < 0) {
                s = {val, len, rowIdx};
                if (++occ * 4 > (int32_t)tab.size() * 3) grow();
                return rowIdx;
            }
            if (s.val == val && s.len == len) return s.id;
            i = (i + 1) & m;
        }
    }
    int32_t lookupOrAssign(uint64_t val, uint32_t len, std::atomic<int32_t>& counter) {
        size_t m = mask(), i = num_hash(val, len) & m;
        for (;;) {
            Slot& s = tab[i];
            if (s.id < 0) {
                int32_t id = counter.fetch_add(1, std::memory_order_relaxed);
                s = {val, len, id};
                if (++occ * 4 > (int32_t)tab.size() * 3) grow();
                return id;
            }
            if (s.val == val && s.len == len) return s.id;
            i = (i + 1) & m;
        }
    }
};

static constexpr uint32_t STRBIT = 0x80000000u;
static constexpr uint32_t DIRBIT = 0x40000000u;

struct LocalParse {
    Interner strs;        // non-numeric node ids (fallback path)
    NumInterner nums;     // numeric node ids (fallback path)
    Interner labels;
    std::vector<int32_t> rowLabel;
    std::vector<uint64_t> rowOff;
    std::vector<uint32_t> nbrs;  // local ids; STRBIT = string id, DIRBIT = direct row id
};

// Parse rows in byte range [a, b); both ends are line-aligned by the caller.
// rowCanon[r] is nonzero iff row r's id is the canonical decimal form of r,
// in which case it holds the id's length (dense-numeric fast path).
static void parseRange(const char* buf, size_t a, size_t b, const uint8_t* rowCanon, size_t R,
                       LocalParse& out) {
    const char* p = buf + a;
    const char* e = buf + b;
    while (p < e) {
        if (*p == '\n' || *p == '\r') {
            ++p;
            continue;
        }
        Span f1, f2, f3;
        if (*p == '"') {
            const char* s = ++p;
            while (p < e && *p != '"') ++p;
            f1 = {s, (uint32_t)(p - s)};
            if (p < e) ++p;
        } else {
            const char* s = p;
            while (p < e && *p != ',' && *p != '\n') ++p;
            f1 = {s, (uint32_t)(p - s)};
        }
        if (p < e && *p == ',') ++p;
        if (p < e && *p == '"') {
            const char* s = ++p;
            while (p < e && *p != '"') ++p;
            f2 = {s, (uint32_t)(p - s)};
            if (p < e) ++p;
        } else {
            const char* s = p;
            while (p < e && *p != ',' && *p != '\n') ++p;
            f2 = {s, (uint32_t)(p - s)};
        }
        if (p < e && *p == ',') ++p;
        if (p < e && *p == '"') {
            const char* s = ++p;
            while (p < e && *p != '"') ++p;
            f3 = {s, (uint32_t)(p - s)};
            if (p < e) ++p;
        } else {
            const char* s = p;
            while (p < e && *p != '\n') ++p;
            f3 = {s, (uint32_t)(p - s)};
            if (f3.len && f3.p[f3.len - 1] == '\r') --f3.len;
        }
        out.rowLabel.push_back(out.labels.intern(f2.p, f2.len, hash_str(f2.p, f2.len)));
        out.rowOff.push_back(out.nbrs.size());
        if (f3.len >= 2 && f3.p[0] == '[') {
            const char* q = f3.p + 1;
            const char* ne = f3.p + f3.len;
            while (ne > q && ne[-1] != ']') --ne;
            if (ne > q) --ne;  // ne points at ']' so the last id span excludes it
            // 4-deep circular pipeline: parse ahead, prefetch slots, intern lagging entry
            Span S[4];
            uint64_t V[4] = {0, 0, 0, 0}, H[4] = {0, 0, 0, 0};
            bool N[4] = {false, false, false, false};
            int len = 0, head = 0;
            if (q < ne) {
                S[0].p = q;
                while (q < ne && *q != ',') ++q;
                S[0].len = (uint32_t)(q - S[0].p);
                if (S[0].len) {
                    N[0] = parse_num(S[0].p, S[0].len, V[0]);
                    if (N[0])
                        __builtin_prefetch(&rowCanon[V[0]]);
                    else
                        H[0] = hash_str(S[0].p, S[0].len),
                        __builtin_prefetch(&out.strs.tab[H[0] & out.strs.mask()]);
                    len = 1;
                }
            }
            auto parseNext = [&]() {
                if (len >= 4 || q >= ne) return;
                ++q;  // skip ','
                if (q >= ne) return;
                int k = (head + len) & 3;
                S[k].p = q;
                while (q < ne && *q != ',') ++q;
                S[k].len = (uint32_t)(q - S[k].p);
                if (!S[k].len) return;
                N[k] = parse_num(S[k].p, S[k].len, V[k]);
                if (N[k])
                    __builtin_prefetch(&rowCanon[V[k]]);  // direct-hit fast path
                else
                    H[k] = hash_str(S[k].p, S[k].len),
                    __builtin_prefetch(&out.strs.tab[H[k] & out.strs.mask()]);
                ++len;
            };
            parseNext();
            parseNext();
            parseNext();
            while (len > 0) {
                parseNext();
                int h = head & 3;
                // fast path: numeric id in canonical decimal form matching its
                // own row -> the global id is exactly the row index
                if (N[h] && V[h] < R && rowCanon[V[h]] == S[h].len) {
                    out.nbrs.push_back(DIRBIT | (uint32_t)V[h]);
                } else if (N[h]) {
                    out.nbrs.push_back((uint32_t)out.nums.intern(V[h], S[h].len));
                } else {
                    out.nbrs.push_back(STRBIT | (uint32_t)out.strs.intern(S[h].p, S[h].len, H[h]));
                }
                ++head;
                --len;
            }
        }
    }
    out.rowOff.push_back(out.nbrs.size());
}

static constexpr int NSH = 256;
static constexpr size_t HLEN = sizeof("node_id,final_label\n") - 1;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input.csv>\n", argv[0]);
        return 1;
    }
    bool timing = getenv("LP_TIMING") != nullptr;
    double t0 = now_s(), t = t0;

    // ---------- map the input into memory ----------
    const char* buf = nullptr;
    size_t F = 0;
    std::vector<char> fbuf;
    int fd = -1;
#if defined(__linux__)
    fd = open(argv[1], O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            F = (size_t)st.st_size;
            void* m = mmap(nullptr, F, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                buf = (const char*)m;
                madvise(m, F, MADV_WILLNEED);
            }
        }
    }
#endif
    if (!buf) {  // fallback: plain read
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        FILE* f = fopen(argv[1], "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", argv[1]);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long fsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        fbuf.resize((size_t)fsz + 2);
        F = fread(fbuf.data(), 1, (size_t)fsz, f);
        fclose(f);
        fbuf[F] = '\n';
        fbuf[F + 1] = '\n';
        buf = fbuf.data();
    }
    if (timing) {
        fprintf(stderr, "map: %.3fs (%.1f MB)\n", now_s() - t, F / 1048576.0);
        t = now_s();
    }

    const int T = omp_get_max_threads();

    // thread byte ranges, aligned forward to the next line boundary
    std::vector<size_t> rng(T + 1);
    for (int i = 0; i <= T; ++i) rng[i] = F * (size_t)i / T;
    for (int i = 1; i < T; ++i) {
        if (rng[i] < F) {
            const void* q = memchr(buf + rng[i], '\n', F - rng[i]);
            rng[i] = q ? (size_t)((const char*)q - buf) + 1 : F;
        }
    }
    for (int i = 1; i < T; ++i)
        if (rng[i] < rng[i - 1]) rng[i] = rng[i - 1];
    {  // thread 0 skips the header line
        const void* q = memchr(buf, '\n', rng[1] > 0 ? rng[1] : (F ? 1 : 0));
        rng[0] = q ? (size_t)((const char*)q - buf) + 1 : F;
    }

    // ---------- pre-pass: count rows, extract id spans and numeric values ----------
    struct RowInfo {
        Span s;
        uint64_t val;
        uint8_t ok;
    };
    std::vector<std::vector<RowInfo>> ptInfo(T);
#pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        const char* p = buf + rng[tid];
        const char* e = buf + rng[tid + 1];
        auto& ri = ptInfo[tid];
        ri.reserve((size_t)(e - p) / 200 + 16);
        while (p < e) {
            while (p < e && (*p == '\r' || *p == '\n')) ++p;
            if (p >= e) break;
            Span f1;
            if (*p == '"') {
                const char* s = ++p;
                while (p < e && *p != '"') ++p;
                f1 = {s, (uint32_t)(p - s)};
                if (p < e) ++p;
            } else {
                const char* s = p;
                while (p < e && *p != ',' && *p != '\n') ++p;
                f1 = {s, (uint32_t)(p - s)};
            }
            uint64_t v = 0;
            bool ok = parse_num(f1.p, f1.len, v);
            ri.push_back({f1, v, static_cast<uint8_t>(ok)});
            const void* q = memchr(p, '\n', (size_t)(e - p));
            if (!q) break;
            p = (const char*)q + 1;
        }
    }
    std::vector<size_t> rowBase(T + 1, 0);
    size_t R = 0;
    for (int i = 0; i < T; ++i) {
        rowBase[i] = R;
        R += ptInfo[i].size();
    }
    std::vector<Span> idSpan(R);
    std::vector<uint64_t> numVal(R, 0);
    std::vector<uint8_t> rowCanon(R, 0);  // nonzero = row id is canonical decimal of its index
    bool allNum = true;
    for (int i = 0; i < T; ++i) {
        const auto& ri = ptInfo[i];
        size_t base = rowBase[i];
        for (size_t j = 0; j < ri.size(); ++j) {
            size_t rowIdx = base + j;
            idSpan[rowIdx] = ri[j].s;
            numVal[rowIdx] = ri[j].val;
            if (!ri[j].ok) {
                allNum = false;
                continue;
            }
            // canonical iff value equals the row index and no leading zeros
            uint64_t v = ri[j].val;
            uint32_t dl = 1;
            for (uint64_t x = v; x >= 10; x /= 10) ++dl;
            rowCanon[rowIdx] =
                (v == rowIdx && ri[j].s.len == dl) ? (uint8_t)dl : (uint8_t)0;
        }
    }
    if (timing) {
        fprintf(stderr, "prepass: %.3fs (R=%zu)\n", now_s() - t, R);
        t = now_s();
    }

    // ---------- parallel parse + fused row-id insertion ----------
    std::vector<LocalParse> locals(T);
    std::vector<std::mutex> numLock(NSH), strLock(NSH);
    std::vector<NumInterner> numSh(NSH);
    std::vector<Interner> strSh(NSH);
    for (int i = 0; i < NSH; ++i) {
        numSh[i].init(R * 2 / NSH + 64);
        strSh[i].init(1024);
    }
#pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        auto& lp = locals[tid];
        size_t a = rng[tid], b = rng[tid + 1];
        size_t est = (b - a) / 300 + 16;
        lp.nums.init(est);
        lp.strs.init(1024);
        lp.labels.init(64);
        lp.rowLabel.reserve(est);
        lp.rowOff.reserve(est + 1);
        lp.nbrs.reserve((b - a) / 8 + 16);
        parseRange(buf, a, b, rowCanon.data(), R, lp);
        // insert this thread's row ids into the sharded global tables (fused)
        const auto& ri = ptInfo[tid];
        for (size_t i = 0; i < ri.size(); ++i) {
            int32_t rowIdx = (int32_t)(rowBase[tid] + i);
            if (ri[i].ok) {
                uint64_t h = num_hash(ri[i].val, ri[i].s.len);
                int sh = (int)(h >> 40 & (NSH - 1));
                std::lock_guard<std::mutex> g(numLock[sh]);
                numSh[sh].insertVal(ri[i].val, ri[i].s.len, rowIdx);
            } else {
                uint64_t h = hash_str(ri[i].s.p, ri[i].s.len);
                int sh = (int)((h >> 40) & (NSH - 1));
                std::lock_guard<std::mutex> g(strLock[sh]);
                strSh[sh].insertVal(ri[i].s.p, ri[i].s.len, h, rowIdx);
            }
        }
    }
    if (timing) {
        fprintf(stderr, "parse: %.3fs\n", now_s() - t);
        t = now_s();
    }

    // ---------- labels ----------
    Interner gLabels;
    gLabels.init(1024);
    std::vector<std::vector<int32_t>> labMap(T);
    for (int tid = 0; tid < T; ++tid) {
        auto& lp = locals[tid];
        labMap[tid].resize(lp.labels.occ);
        for (const auto& en : lp.labels.tab)
            if (en.used) labMap[tid][en.id] = gLabels.intern(en.p, en.len, en.h);
    }
    const int32_t L = gLabels.occ;

    std::vector<Span> labSpan(L);
    for (const auto& e : gLabels.tab)
        if (e.used) labSpan[e.id] = {e.p, e.len};
    std::vector<int32_t> orderByRank(L);
    std::iota(orderByRank.begin(), orderByRank.end(), 0);
    std::sort(orderByRank.begin(), orderByRank.end(), [&](int32_t a, int32_t b) {
        Span A = labSpan[a], B = labSpan[b];
        int c = memcmp(A.p, B.p, std::min(A.len, B.len));
        if (c) return c < 0;
        return A.len < B.len;
    });
    std::vector<int32_t> rank(L);
    for (int32_t r = 0; r < L; ++r) rank[orderByRank[r]] = r;

    // ---------- fused: resolve local ids + build CSR ----------
    // Extra ids (neighbour-only, >= R) have degree 0, so the degree prefix over
    // rows is already final for adjacency; extend it for the extra range below.
    std::vector<uint64_t> offRow(R + 1, 0);
    for (int tid = 0; tid < T; ++tid) {
        auto& lp = locals[tid];
        for (size_t i = 0; i < lp.rowLabel.size(); ++i)
            offRow[rowBase[tid] + (int32_t)i + 1] = lp.rowOff[i + 1] - lp.rowOff[i];
    }
    for (int32_t i = 0; i < (int32_t)R; ++i) offRow[i + 1] += offRow[i];
    auto adj = std::make_unique_for_overwrite<uint32_t[]>(offRow[R]);
#if defined(__linux__)
    madvise(adj.get(), offRow[R] * sizeof(uint32_t), MADV_HUGEPAGE);
#endif
    std::atomic<int32_t> extraCnt((int32_t)R);  // neighbour-only ids come after rows
    std::vector<std::vector<int32_t>> nodeMapNum(T), nodeMapStr(T);
#pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        auto& lp = locals[tid];
        // resolve numeric fallback ids
        nodeMapNum[tid].resize(lp.nums.occ);
        const size_t kn = lp.nums.idkey.size();
        for (size_t k = 0; k < kn; ++k) {
            if (k + 2 < kn) {
                auto [v2, l2] = lp.nums.idkey[k + 2];
                uint64_t h2 = num_hash(v2, l2);
                int sh2 = (int)(h2 >> 40 & (NSH - 1));
                __builtin_prefetch(&numSh[sh2].tab[h2 & numSh[sh2].mask()]);
            }
            auto [val, len] = lp.nums.idkey[k];
            int sh = (int)(num_hash(val, len) >> 40 & (NSH - 1));
            std::lock_guard<std::mutex> g(numLock[sh]);
            nodeMapNum[tid][k] = numSh[sh].lookupOrAssign(val, len, extraCnt);
        }
        // resolve string fallback ids
        nodeMapStr[tid].resize(lp.strs.occ);
        const size_t nTab = lp.strs.tab.size();
        for (size_t i = 0; i < nTab; ++i) {
            const Interner::Ent& en = lp.strs.tab[i];
            if (!en.used) continue;
            int sh = (int)((en.h >> 40) & (NSH - 1));
            std::lock_guard<std::mutex> g(strLock[sh]);
            nodeMapStr[tid][en.id] = strSh[sh].lookupOrAssign(en.p, en.len, en.h, extraCnt);
        }
        // CSR copy (depends only on this thread's node maps)
        uint32_t* dst = adj.get() + offRow[rowBase[tid]];
        const int32_t* nmN = nodeMapNum[tid].data();
        const int32_t* nmS = nodeMapStr[tid].data();
        for (uint32_t v : lp.nbrs) {
            if (v & STRBIT)
                *dst++ = (uint32_t)nmS[v & 0x7fffffffu];
            else if (v & DIRBIT)
                *dst++ = v & 0x3fffffffu;
            else
                *dst++ = (uint32_t)nmN[v];
        }
    }
    const int32_t N = extraCnt.load(std::memory_order_relaxed);
    // off[] over all N nodes (extra nodes have zero degree)
    std::vector<uint64_t> off(N + 1, 0);
    std::copy(offRow.begin(), offRow.end(), off.begin());
    for (int32_t i = R + 1; i <= N; ++i) off[i] = offRow[R];
    if (timing) {
        fprintf(stderr, "resolve+csr: %.3fs (N=%d, extra=%d)\n", now_s() - t, N,
                N - (int32_t)R);
        t = now_s();
    }

    // ---------- initial labels ----------
    std::vector<int32_t> cur(N, 0), nxt(N, 0);
#if defined(__linux__)
    madvise(cur.data(), (size_t)N * sizeof(int32_t), MADV_HUGEPAGE);
    madvise(nxt.data(), (size_t)N * sizeof(int32_t), MADV_HUGEPAGE);
#endif
    {
        std::vector<int32_t> rowLabelG(R);
#pragma omp parallel num_threads(T)
        {
            int tid = omp_get_thread_num();
            auto& lp = locals[tid];
            size_t base = rowBase[tid];
            for (size_t i = 0; i < lp.rowLabel.size(); ++i)
                rowLabelG[base + i] = labMap[tid][lp.rowLabel[i]];
        }
        for (size_t r = 0; r < R; ++r) cur[r] = rank[rowLabelG[r]];
    }
    {
        std::vector<LocalParse>().swap(locals);
        std::vector<std::vector<int32_t>>().swap(nodeMapNum);
        std::vector<std::vector<int32_t>>().swap(nodeMapStr);
        std::vector<std::vector<int32_t>>().swap(labMap);
        std::vector<std::vector<RowInfo>>().swap(ptInfo);
    }

    // ---------- synchronous label propagation (active-frontier) ----------
    int iters = 0;
    {
        std::vector<std::vector<uint32_t>> stamp(T, std::vector<uint32_t>(L, 0));
        std::vector<std::vector<int32_t>> cnt(T, std::vector<int32_t>(L, 0));
        std::vector<uint32_t> stampVal(T, 0);
        std::vector<int32_t> active(N), nextActive(N);
        std::vector<std::atomic<uint8_t>> mark(N);
        for (int32_t i = 0; i < N; ++i) active[i] = i;
        size_t na = (size_t)N;
        std::vector<std::vector<int32_t>> chg(T), sched(T);
        int64_t changed = 1;
        bool frontierMode = false;
        while (changed) {
            changed = 0;
            ++iters;
            for (int i = 0; i < T; ++i) chg[i].clear();
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                uint32_t* st = stamp[tid].data();
                int32_t* cn = cnt[tid].data();
                uint32_t& sv = stampVal[tid];
                auto nodeBody = [&](int32_t u) {
                    ++sv;
                    int32_t best = -1, bc = 0;
                    uint64_t jEnd = off[u + 1];
                    for (uint64_t j = off[u]; j < jEnd; ++j) {
                        if (j + 24 < jEnd) __builtin_prefetch(&cur[adj[j + 24]]);
                        int32_t l = cur[adj[j]];
                        if (st[l] != sv) {
                            st[l] = sv;
                            cn[l] = 1;
                        } else {
                            ++cn[l];
                        }
                        int32_t c = cn[l];
                        if (c > bc || (c == bc && l < best)) {
                            bc = c;
                            best = l;
                        }
                    }
                    if (best < 0) return;  // degree-0: keep label
                    nxt[u] = best;
                    if (best != cur[u]) chg[tid].push_back(u);
                };
                if (frontierMode) {
                    size_t a0 = na * (size_t)tid / T, a1 = na * (size_t)(tid + 1) / T;
                    for (size_t i = a0; i < a1; ++i) nodeBody(active[i]);
                } else {
#pragma omp for schedule(dynamic, 16384)
                    for (int64_t u = 0; u < N; ++u) nodeBody((int32_t)u);
                }
            }
            for (int i = 0; i < T; ++i) changed += (int64_t)chg[i].size();
            if (!changed) break;
            // commit new labels for changed nodes (synchronous semantics preserved:
            // all reads used cur, writes go to nxt, cur updated only after the round)
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                for (int32_t u : chg[tid]) cur[u] = nxt[u];
            }
            // schedule: only nodes adjacent to a changed node can change next round
            if (changed * 4 < (int64_t)N) {
                frontierMode = true;
                for (int i = 0; i < T; ++i) sched[i].clear();
#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    for (int32_t u : chg[tid]) {
                        for (uint64_t j = off[u]; j < off[u + 1]; ++j) {
                            int32_t v = (int32_t)adj[j];
                            if (!mark[v].exchange(1, std::memory_order_relaxed))
                                sched[tid].push_back(v);
                        }
                    }
                }
                size_t k = 0;
                for (int i = 0; i < T; ++i) {
                    for (int32_t v : sched[i]) nextActive[k++] = v;
                    sched[i].clear();
                    sched[i].shrink_to_fit();
                }
                for (size_t i = 0; i < k; ++i) mark[nextActive[i]].store(0, std::memory_order_relaxed);
                std::swap(active, nextActive);
                na = k;
            } else if (frontierMode) {
                frontierMode = false;
                for (int32_t i = 0; i < N; ++i) active[i] = i;
                na = (size_t)N;
            }
        }
    }
    if (timing) {
        fprintf(stderr, "iterate: %.3fs (%d iterations)\n", now_s() - t, iters);
        t = now_s();
    }

    // ---------- output order ----------
    std::vector<int32_t> rows(R);
    std::iota(rows.begin(), rows.end(), 0);

    auto spanLess = [&](Span A, Span B) {
        int c = memcmp(A.p, B.p, std::min(A.len, B.len));
        return c ? c < 0 : A.len < B.len;
    };
    bool sortedAlready;
    if (allNum) {
        sortedAlready = true;
        for (size_t r = 1; r < R; ++r)
            if (numVal[r - 1] > numVal[r]) {
                sortedAlready = false;
                break;
            }
    } else {
        sortedAlready = true;
        for (size_t r = 1; r < R; ++r)
            if (spanLess(idSpan[r], idSpan[r - 1])) {
                sortedAlready = false;
                break;
            }
    }

    if (!sortedAlready) {
        if (allNum) {  // LSD radix on the numeric key, stable
            uint64_t mx = 0;
            for (size_t r = 0; r < R; ++r) mx = std::max(mx, numVal[r]);
            int nb = 1;
            while (mx >> 8) {
                mx >>= 8;
                ++nb;
            }
            struct KV {
                uint64_t v;
                int32_t r;
                int32_t pad;
            };
            std::vector<KV> a(R), b(R);
            for (size_t r = 0; r < R; ++r) a[r] = {numVal[r], (int32_t)r, 0};
            std::vector<KV> *src = &a, *dst = &b;
            size_t counts[256];
            for (int byte = 0; byte < nb; ++byte) {
                memset(counts, 0, sizeof(counts));
                for (size_t i = 0; i < R; ++i) ++counts[((*src)[i].v >> (byte * 8)) & 0xFF];
                size_t sum = 0;
                for (int i = 0; i < 256; ++i) {
                    size_t c2 = counts[i];
                    counts[i] = sum;
                    sum += c2;
                }
                for (size_t i = 0; i < R; ++i)
                    (*dst)[counts[((*src)[i].v >> (byte * 8)) & 0xFF]++] = (*src)[i];
                std::swap(src, dst);
            }
            for (size_t i = 0; i < R; ++i) rows[i] = (*src)[i].r;
        } else {
            std::sort(rows.begin(), rows.end(),
                      [&](int32_t a2, int32_t b2) { return spanLess(idSpan[a2], idSpan[b2]); });
        }
    }
    if (timing) {
        fprintf(stderr, "sort: %.3fs\n", now_s() - t);
        t = now_s();
    }

    // ---------- build and write output (parallel) ----------
    std::vector<uint32_t> rowLen(R);
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < (int64_t)R; ++r)
        rowLen[r] = idSpan[r].len + labSpan[orderByRank[cur[r]]].len + 2;
    std::vector<uint64_t> rowOut(R + 1);
    rowOut[0] = HLEN;
    for (size_t i = 0; i < R; ++i) rowOut[i + 1] = rowOut[i] + rowLen[rows[i]];
    auto out = std::make_unique_for_overwrite<char[]>(rowOut[R]);
    memcpy(out.get(), "node_id,final_label\n", HLEN);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)R; ++i) {
        int32_t r = rows[i];
        char* w = out.get() + rowOut[i];
        Span id = idSpan[r];
        Span lb = labSpan[orderByRank[cur[r]]];
        memcpy(w, id.p, id.len);
        w += id.len;
        *w++ = ',';
        memcpy(w, lb.p, lb.len);
        w += lb.len;
        *w = '\n';
    }
    FILE* g = fopen("output.csv", "wb");
    if (!g) {
        fprintf(stderr, "cannot write output.csv\n");
        return 1;
    }
    fwrite(out.get(), 1, rowOut[R], g);
    fclose(g);
    if (fd >= 0) {
        munmap((void*)buf, F);
        close(fd);
    }
    if (timing)
        fprintf(stderr, "write: %.3fs (%.1f MB)\n", now_s() - t, rowOut[R] / 1048576.0);
    return 0;
}
