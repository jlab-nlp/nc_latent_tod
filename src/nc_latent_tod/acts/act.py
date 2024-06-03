import json
import logging
from typing import Tuple, Generator, Any, Dict


class Entity:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if kwargs:
            for key, value in kwargs.items():
                if value is not None:  # treat None as unset, distinct from other falsy values
                    setattr(self, key, value)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        # slow, but should work. Depends on serialization being accurate
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in vars(self).items():
            if isinstance(v, (Entity, Act)):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return {
            "__type": self.__class__.__name__,
            **d
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], loading_context: Dict[str, Any] = None) -> "Entity":
        loading_context = loading_context or globals()
        loading_context['Entity'] = Entity
        loading_context['Act'] = Act
        data = data.copy()
        for key, value in data.items():
            if isinstance(value, dict) and "__type" in value:
                data_cls = loading_context.get(value["__type"], None)
                if data_cls is None:
                    logging.error(f"Unknown type {value['__type']}, falling back to Entity")
                    data_cls = Entity
                data[key] = data_cls.from_dict(value, loading_context=loading_context)
        return cls(**{k: v for k, v in data.items() if k != '__type'})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class Act(Entity):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if kwargs:
            for key, value in kwargs.items():
                if value is not None:  # treat None as unset, distinct from other falsy values
                    setattr(self, key, value)

    def to_act_tuples(self) -> Generator[Tuple[str, str, str, str], None, None]:
        act: str = self.__class__.__name__.lower()
        service: str = getattr(self, "service", None)
        for name, value in vars(self).items():
            if name == "service":
                continue
            yield act, service, name, value
