import abc, dataclassy
from dataclassy.dataclass import DataClassMeta

class DataClassTransformArgsMeta(DataClassMeta):
  """
  allows you to define a __init__ function in the dataclass.
  the dataclass __init__ will still be generated, and you
  can access it via super().__init__.
  """
  def __call__(cls, *args, **kwargs):
    newargs, newkwargs = cls.transforminitargs(*args, **kwargs)
    return super().__call__(*newargs, **newkwargs)

@dataclassy.dataclass(meta=DataClassTransformArgsMeta)
class DataClassTransformArgs:
  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    return args, kwargs

class MetaDataAnnotation:
  def __init__(self, typ, **kwargs):
    if isinstance(typ, MetaDataAnnotation):
      raise TypeError("You don't want nested MetaDataAnnotations")
    self.type = typ
    self.metadata = kwargs

class DataClassWithMetaDataMeta(DataClassMeta):
  def __new__(mcs, name, bases, dict_, **kwargs):
    dataclass_bases = [vars(b) for b in bases if hasattr(b, '__annotationmetadata__')]
    __annotationmetadata__ = {}
    for b in dataclass_bases + [dict_]:
      __annotationmetadata__.update(b.get("__annotationmetadata__", {}))
    dict_["__annotationmetadata__"] = __annotationmetadata__

    __annotations__ = dict_.setdefault("__annotations__", {})
    for name, typ in __annotations__.items():
      if isinstance(typ, MetaDataAnnotation):
        __annotations__[name] = typ.type
        __annotationmetadata__[name] = typ.metadata

    return super().__new__(mcs, name, bases, dict_, **kwargs)

@dataclassy.dataclass(meta=DataClassWithMetaDataMeta)
class DataClassWithMetaData:
  @classmethod
  def metadata(cls, fieldname):
    return cls.__annotationmetadata__.get(fieldname, {})

class MyDataClassMeta(abc.ABCMeta, DataClassTransformArgsMeta, DataClassWithMetaDataMeta):
  pass

class MyDataClass(DataClassTransformArgs, DataClassWithMetaData, metaclass=MyDataClassMeta):
  pass
