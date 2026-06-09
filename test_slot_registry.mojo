from std.memory import Span

from continuous_batching.slot_registry import SlotRegistry


def check(cond: Bool, msg: String) -> Int:
    if cond:
        print("  ok  -", msg)
        return 0
    print("  FAIL-", msg)
    return 1


def seq(read xs: List[Int32]) -> Span[Int32, origin_of(xs)]:
    return Span(xs)


def test_match_on_empty() -> Int:
    print("match_on_empty")
    var f = 0
    var reg = SlotRegistry(4)
    var q: List[Int32] = [1, 2, 3]
    f += check(reg.match_append(seq(q)) == -1, "no resident slots -> no match")
    f += check(reg.lru_victim() == -1, "no resident slots -> no victim")
    return f


def test_append_match() -> Int:
    print("append_match")
    var f = 0
    var reg = SlotRegistry(4)
    var prefix: List[Int32] = [1, 2, 3]
    reg.open(0, owner_id=10, now=UInt(100))
    reg.extend(0, seq(prefix), 0, 0, 3, UInt(100))
    reg.set_warm(0)

    var cont: List[Int32] = [1, 2, 3, 4, 5]
    f += check(reg.match_append(seq(cont)) == 0, "strict-prefix continuation matches")
    f += check(reg.length(0) == 3, "history length tracks fed tokens")

    var other: List[Int32] = [9, 9, 9, 9]
    f += check(reg.match_append(seq(other)) == -1, "non-prefix request does not match")

    var exact: List[Int32] = [1, 2, 3]
    f += check(reg.match_append(seq(exact)) == 0, "exact-length match reuses the slot for regeneration")
    return f


def test_owner_excluded() -> Int:
    print("owner_excluded")
    var f = 0
    var reg = SlotRegistry(4)
    var prefix: List[Int32] = [7, 8]
    reg.open(2, owner_id=5, now=UInt(50))
    reg.extend(2, seq(prefix), 0, 0, 2, UInt(50))

    var cont: List[Int32] = [7, 8, 9]
    f += check(reg.match_append(seq(cont)) == -1, "a live (owned) slot is not reusable")
    f += check(reg.lru_victim() == -1, "a live slot is never an LRU victim")

    reg.set_warm(2)
    f += check(reg.match_append(seq(cont)) == 2, "set_warm makes it reusable")
    f += check(reg.lru_victim() == 2, "set_warm makes it evictable")
    return f


def test_longest_prefix_wins() -> Int:
    print("longest_prefix_wins")
    var f = 0
    var reg = SlotRegistry(4)
    var short: List[Int32] = [1, 2]
    var long: List[Int32] = [1, 2, 3, 4]
    reg.open(0, owner_id=1, now=UInt(10))
    reg.extend(0, seq(short), 0, 0, 2, UInt(10))
    reg.set_warm(0)
    reg.open(1, owner_id=2, now=UInt(20))
    reg.extend(1, seq(long), 0, 0, 4, UInt(20))
    reg.set_warm(1)

    var cont: List[Int32] = [1, 2, 3, 4, 5, 6]
    f += check(reg.match_append(seq(cont)) == 1, "longest matching prefix wins")
    return f


def test_lru_victim() -> Int:
    print("lru_victim")
    var f = 0
    var reg = SlotRegistry(4)
    var a: List[Int32] = [1]
    var b: List[Int32] = [2]
    var c: List[Int32] = [3]
    reg.open(0, owner_id=1, now=UInt(300))
    reg.extend(0, seq(a), 0, 0, 1, UInt(300))
    reg.set_warm(0)
    reg.open(1, owner_id=2, now=UInt(100))
    reg.extend(1, seq(b), 0, 0, 1, UInt(100))
    reg.set_warm(1)
    reg.open(2, owner_id=3, now=UInt(200))
    reg.extend(2, seq(c), 0, 0, 1, UInt(200))
    reg.set_warm(2)

    f += check(reg.lru_victim() == 1, "oldest last_used is the victim")

    reg.close(1)
    f += check(reg.lru_victim() == 2, "after eviction next-oldest is the victim")
    f += check(reg.length(1) == 0, "close drops history")
    var cont: List[Int32] = [2, 2]
    f += check(reg.match_append(seq(cont)) == -1, "closed slot no longer matches")
    return f


def main():
    var failures = 0
    failures += test_match_on_empty()
    failures += test_append_match()
    failures += test_owner_excluded()
    failures += test_longest_prefix_wins()
    failures += test_lru_victim()
    print()
    if failures == 0:
        print("all slot-registry checks passed")
    else:
        print(failures, "check(s) FAILED")
