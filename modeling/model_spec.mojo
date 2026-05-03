from std.memory import UnsafePointer


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2

struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4


comptime DEFAULT_ALIGNMENT = 64


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


trait ShapeLike:
    comptime GLOBAL_N: Int
    comptime GLOBAL_M: Int
    comptime DATA_N: Int
    comptime DATA_M: Int
    comptime N: Int
    comptime M: Int

    @staticmethod
    def bytes[E: Encoding]() -> Int: ...

struct Shape[
    global_n: Int, global_m: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    degree: Int = 1,
    align_n: Int = 1, align_m: Int = 1,
](ShapeLike):
    comptime GLOBAL_N = Self.global_n
    comptime GLOBAL_M = Self.global_m
    comptime DATA_N = Self.global_n // Self.degree if Self.shard_n else Self.global_n
    comptime DATA_M = Self.global_m // Self.degree if Self.shard_m else Self.global_m
    comptime N = align_up(Self.DATA_N, Self.align_n)
    comptime M = align_up(Self.DATA_M, Self.align_m)

    @staticmethod
    def bytes[E: Encoding]() -> Int:
        return Self.N * Self.M * E.ELEMENT_BYTES


trait DistributionDegreeLike:
    comptime DEGREE: Int
    comptime TENSOR: Int
    comptime CONTEXT: Int
    comptime EXPERT: Int
    comptime VOCAB: Int


struct DistributionDegree[degree: Int](DistributionDegreeLike):
    comptime DEGREE = Self.degree
    comptime TENSOR = Self.degree
    comptime CONTEXT = Self.degree
    comptime EXPERT = Self.degree
    comptime VOCAB = Self.degree


comptime TensorParallelRows[n: Int, D: DistributionDegreeLike]: Int = n // D.TENSOR
comptime TensorParallelColumns[m: Int, D: DistributionDegreeLike]: Int = m // D.TENSOR
comptime ContextParallelRows[n: Int, D: DistributionDegreeLike]: Int = n // D.CONTEXT
comptime ExpertsPerRank[experts: Int, D: DistributionDegreeLike]: Int = experts // D.EXPERT
comptime ExpertParallelRows[
    experts: Int, rows_per_expert: Int, D: DistributionDegreeLike,
]: Int = ExpertsPerRank[experts, D] * rows_per_expert
comptime VocabularyParallelRows[n: Int, D: DistributionDegreeLike]: Int = n // D.VOCAB


comptime Replicated[n: Int, m: Int] = Shape[
    n, m, shard_n=False, shard_m=False, degree=1,
]
comptime TensorRowSharded[n: Int, m: Int, D: DistributionDegreeLike] = Shape[
    n, m, shard_n=True, degree=D.TENSOR,
]
comptime TensorColumnSharded[n: Int, m: Int, D: DistributionDegreeLike] = Shape[
    n, m, shard_m=True, degree=D.TENSOR,
]
comptime ContextRowSharded[n: Int, m: Int, D: DistributionDegreeLike] = Shape[
    n, m, shard_n=True, degree=D.CONTEXT,
]
comptime ExpertRowBlockSharded[
    experts: Int, rows_per_expert: Int, cols: Int, D: DistributionDegreeLike,
] = Shape[
    experts * rows_per_expert, cols, shard_n=True, degree=D.EXPERT,
]
comptime VocabularyRowSharded[n: Int, m: Int, D: DistributionDegreeLike] = Shape[
    n, m, shard_n=True, degree=D.VOCAB,
]


comptime DISTRIBUTED = -1
comptime HOST_RANK = 0


@fieldwise_init
struct StaticView[E: Encoding, S: ShapeLike]:
    comptime DTYPE = Self.E.DTYPE
    var ptr: UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]

    @always_inline
    def as_ptr[dtype: DType = Self.DTYPE](self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](self.ptr)


@fieldwise_init
struct WeightDesc(Copyable):
    var name: String
    var arena_offset: Int
    var dtype: DType
    var element_bytes: Int
    var global_rows: Int
    var global_cols: Int
    var local_cols: Int
    var data_rows: Int
    var data_cols: Int
    var target_rank: Int
