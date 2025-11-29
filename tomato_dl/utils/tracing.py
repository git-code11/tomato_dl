import typing as tp
import enum
import operator
import functools


class AccessorVariant(enum.Enum):
    Sequence = 1
    Mapping = 2
    Attribute = 3


class Accessor(tp.NamedTuple):
    variant: AccessorVariant
    value: tp.Any

    def __call__(self, obj: tp.Any) -> tp.Any:
        if self.variant is AccessorVariant.Sequence:
            return operator.itemgetter(self.value)(obj)
        elif self.variant is AccessorVariant.Mapping:
            return operator.itemgetter(self.value)(obj)
        elif self.variant is AccessorVariant.Attribute:
            return operator.attrgetter(self.value)(obj)
        return obj

    def update(self, obj: tp.Any, new_obj: tp.Any) -> tp.NoReturn:
        if self.variant is AccessorVariant.Sequence:
            obj[self.value] = new_obj
        elif self.variant is AccessorVariant.Mapping:
            obj[self.value] = new_obj
        elif self.variant is AccessorVariant.Attribute:
            setattr(obj, self.value, new_obj)

    @classmethod
    def sequence(cls, value: int) -> tp.Self:
        return cls(AccessorVariant.Sequence, value)

    @classmethod
    def mapping(cls, value: str) -> tp.Self:
        return cls(AccessorVariant.Mapping, value)

    @classmethod
    def attribute(cls, value: str) -> tp.Self:
        return cls(AccessorVariant.Attribute, value)

    def __str__(self) -> str:
        if self.variant is AccessorVariant.Sequence:
            return f"[{self.value}]"
        elif self.variant is AccessorVariant.Mapping:
            return f"['{self.value}']"
        elif self.variant is AccessorVariant.Attribute:
            return f".{self.value}"
        return self.value


class TracerNode(tp.NamedTuple):
    accessors: list[Accessor]

    def __str__(self) -> str:
        return str.join('', map(str, self.accessors))

    def __call__(self, obj: tp.Any) -> tp.Any:
        return self.get(obj, self.accessors)

    @classmethod
    def get(cls, obj, accessors: list[Accessor]) -> tp.Any:
        return functools.reduce(lambda acc, x: x(acc), accessors, obj)

    def update(self, obj: tp.Any, new_obj: tp.Any) -> tp.Any:
        if len(self.accessors) == 0:
            raise Exception("Not Nested")
        selected = self.get(obj, self.accessors[:-1])
        if selected is None:
            raise Exception("Parent Object is None")
        self.accessors[-1].update(selected, new_obj)


class Tracer:
    """
      Tracer is an utility that helps extract node address recursively
      matching a defined condition

      py```
      model = ...
      nodes = Tracer.list_layers(
              model,
              lambda x: isinstance(x, keras.layers.Layer)
          )
      # Print all matched nodes with address respectively
      for node in nodes:
          print(f"{node} => node(model)")
      ````

    """

    @classmethod
    def list_layers(cls, obj: tp.Any, is_node: tp.Callable[[tp.Any], bool], prefix: list[Accessor] = []) -> list[list[Accessor]]:
        tree = list()
        if is_node(obj):
            tree.append(TracerNode(prefix))
            for var in vars(obj):
                if str.startswith(var, '_') or str.startswith(var, '__'):
                    # Skip magic variable
                    continue
                else:
                    item = getattr(obj, var)
                    output = cls.list_layers(
                        item, [*prefix, Accessor.attribute(var)])
                    tree = tree + output

        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                output = cls.list_layers(
                    item, [*prefix, Accessor.sequence(idx)])
                tree = tree + output

        elif isinstance(obj, dict):
            for key, item in obj.items():
                output = cls.list_layers(
                    item, [*prefix, Accessor.mapping(key)])
                tree = tree + output

        return tree


class TraceCaller:
    """
        cloned = keras.models.clone_model(model)

        trace = TraceCaller(cloned, node_2)

        trace.trace_output = True
        trace.trace_input = True

        trace.attach()
        trace.attached

        cloned(input_data)

        print(trace.inputs)
        print(trace.outputs)

        trace.reset()
        print(trace.outputs)

        trace.detach()
        trace.attached
    """

    def __init__(self, parent_obj: tp.Any, tracer: TracerNode, *,
                 trace_input: bool = False, trace_output: bool = False):
        self._attached = False
        self.caller = None
        self.parent_obj = parent_obj
        self.tracer = tracer
        self.trace_input = trace_input
        self.trace_output = trace_output
        self.reset()

    # To ensure we can get value from caller when attached
    def __getattr__(self, *args, **kwargs):
        if self._attached:
            return self.caller.__getattr__(*args, **kwargs)
        return super().__getattr__(*args, **kwargs)

    def reset(self):
        self._inputs = None
        self._outputs = None

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def outputs(self) -> tp.Optional[tp.Any]:
        return self._outputs

    @property
    def inputs(self) -> tp.Optional[tp.Any]:
        return self._inputs

    def __call__(self, *args, **kwargs) -> tp.Optional[tp.Any]:
        if self.trace_input:
            self._inputs = dict(args=args, kwargs=kwargs)
        outputs = self.caller(*args, **kwargs)
        if self.trace_output:
            self._outputs = outputs
        return outputs

    def attach(self):
        if self._attached:
            return
        self.caller = self.tracer(self.parent_obj)
        self.tracer.update(self.parent_obj, self)
        self._attached = True

    def detach(self):
        if not self._attached:
            return
        self.tracer.update(self.parent_obj, self.caller)
        self.caller = None
        self._attached = False
