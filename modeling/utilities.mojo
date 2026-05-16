from std.reflection import reflect


def _default_value[T: Defaultable & ImplicitlyDestructible]() -> T:
    return T()


trait FieldwiseDefault(Defaultable):
    def __init__(out self):
        comptime for i in range(reflect[Self].field_count()):
            comptime FT = reflect[Self].field_types()[i]
            comptime if conforms_to(FT, Defaultable & ImplicitlyDestructible):
                reflect[Self].field_ref[i](self) = _default_value[FT]()
            else:
                comptime assert False, "field is not Defaultable"
