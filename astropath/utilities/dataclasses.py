import abc
from dataclassy import dataclass
from dataclassy.dataclass import DataClassMeta

class DataClassTransformArgsMeta(DataClassMeta):
  def __call__(cls, *args, **kwargs):
    newargs, newkwargs = cls.transforminitargs(*args, **kwargs)
    return super().__call__(*newargs, **newkwargs)

class DataClassTransformArgsBase(metaclass=DataClassTransformArgsMeta):
  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    return args, kwargs

@dataclass(meta=DataClassTransformArgsMeta)
class DataClassTransformArgs(DataClassTransformArgsBase): pass

@dataclass(meta=DataClassTransformArgsMeta, frozen=True)
class DataClassTransformArgsFrozen(DataClassTransformArgsBase): pass

class MetaDataAnnotation:
  __notgiven = object()
  def __init__(self, defaultvalue=__notgiven, **kwargs):
    if defaultvalue is not self.__notgiven:
      self.defaultvalue = defaultvalue
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
      annovalue = dict_.get(annoname, None)
      if isinstance(typ, MetaDataAnnotation):
        raise TypeError(f"MetaDataAnnotation should be given as the default value for {annoname}, not the type")
      if isinstance(annovalue, MetaDataAnnotation):
        try:
          dict_[annoname] = annovalue.defaultvalue
        except AttributeError:
          del dict_[annoname]
        __annotationmetadata__[annoname] = annovalue.metadata

    return super().__new__(mcs, name, bases, dict_, **kwargs)

class DataClassWithMetaDataBase:
  @classmethod
  def metadata(cls, fieldname):
    return cls.__annotationmetadata__.get(fieldname, {})

@dataclass(meta=DataClassWithMetaDataMeta)
class DataClassWithMetaData(DataClassWithMetaDataBase): pass

@dataclass(meta=DataClassWithMetaDataMeta, frozen=True)
class DataClassWithMetaDataFrozen(DataClassWithMetaDataBase): pass

class MyDataClassMeta(abc.ABCMeta, DataClassTransformArgsMeta, DataClassWithMetaDataMeta):
  pass

class MyDataClassBase(DataClassTransformArgsBase, DataClassWithMetaDataBase): pass

class MyDataClass(MyDataClassBase, DataClassTransformArgs, DataClassWithMetaData, metaclass=MyDataClassMeta):
  def __post_init__(self, *, readingfromfile=False):
    pass

class MyDataClassFrozen(MyDataClassBase, DataClassTransformArgsFrozen, DataClassWithMetaDataFrozen, metaclass=MyDataClassMeta):
  def __post_init__(self, *, readingfromfile=False):
    pass
