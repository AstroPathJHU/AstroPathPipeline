import abc, dataclassy
from dataclassy.dataclass import DataClassMeta

class DataClassTransformArgsMeta(DataClassMeta):
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
    for annoname, typ in __annotations__.items():
      if isinstance(typ, MetaDataAnnotation):
        __annotations__[annoname] = typ.type
        __annotationmetadata__[annoname] = typ.metadata

    return super().__new__(mcs, name, bases, dict_, **kwargs)

@dataclassy.dataclass(meta=DataClassWithMetaDataMeta)
class DataClassWithMetaData:
  @classmethod
  def metadata(cls, fieldname):
    return cls.__annotationmetadata__.get(fieldname, {})

class DataClassSuperInitMeta(DataClassMeta):
  def __new__(mcs, name, bases, dict_, **kwargs):
    if "__init__" not in dict_:
      def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
      dict_["__init__"] = __init__
    return super().__new__(mcs, name, bases, dict_, **kwargs)

class MyDataClassMeta(abc.ABCMeta, DataClassTransformArgsMeta, DataClassWithMetaDataMeta, DataClassSuperInitMeta):
  pass

class MyDataClass(DataClassTransformArgs, DataClassWithMetaData, metaclass=MyDataClassMeta):
  pass
