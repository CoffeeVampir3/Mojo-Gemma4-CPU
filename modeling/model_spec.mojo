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

struct I8(Encoding):
    comptime DTYPE = DType.int8
    comptime ELEMENT_BYTES = 1


comptime DEFAULT_ALIGNMENT = 64


@always_inline
def align_up(value: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


trait ShapeLike:
    comptime SIZE_ON_DISK_N: Int
    comptime SIZE_ON_DISK_M: Int
    comptime LOGICAL_N: Int
    comptime LOGICAL_M: Int
    comptime DATA_N: Int
    comptime DATA_M: Int

    @staticmethod
    def bytes[E: Encoding]() -> Int: ...

struct Shape[
    rows_on_disk: Int, cols_on_disk: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    degree: Int = 1,
    block_n: Int = 1, block_m: Int = 1,
](ShapeLike):
    comptime SIZE_ON_DISK_N = Self.rows_on_disk
    comptime SIZE_ON_DISK_M = Self.cols_on_disk
    comptime LOGICAL_N = (
        align_up(Self.rows_on_disk, Self.degree * Self.block_n)
        if Self.shard_n else Self.rows_on_disk
    )
    comptime LOGICAL_M = (
        align_up(Self.cols_on_disk, Self.degree * Self.block_m)
        if Self.shard_m else Self.cols_on_disk
    )
    comptime DATA_N = Self.LOGICAL_N // Self.degree if Self.shard_n else Self.LOGICAL_N
    comptime DATA_M = Self.LOGICAL_M // Self.degree if Self.shard_m else Self.LOGICAL_M

    @staticmethod
    def bytes[E: Encoding]() -> Int:
        return Self.DATA_N * Self.DATA_M * E.ELEMENT_BYTES


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


comptime Replicated[n: Int, m: Int] = Shape[
    n, m, shard_n=False, shard_m=False, degree=1,
]
comptime TensorRowSharded[
    n: Int, m: Int, D: DistributionDegreeLike, block: Int = 1,
] = Shape[
    n, m, shard_n=True, degree=D.TENSOR, block_n=block,
]
comptime TensorColumnSharded[
    n: Int, m: Int, D: DistributionDegreeLike, block: Int = 1,
] = Shape[
    n, m, shard_m=True, degree=D.TENSOR, block_m=block,
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
