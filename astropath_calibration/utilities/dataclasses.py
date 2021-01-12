import abc, dataclassy
from dataclassy.dataclass import DataClassInit, DataClassMeta

class DataClassSuperNewMeta(DataClassMeta):
  """
  allows you to define a __new__ function in the dataclass.
  the dataclass __new__ will still be generated, and you
  can access it via super().__new__.
  """
  def __new__(mcs, name, bases, dict_, *, _calledfromnew=False, **kwargs):
    if _calledfromnew or "__new__" not in dict_:
      return super().__new__(mcs, name, bases, dict_, **kwargs)
    new = dict_.pop("__new__")
    cls = super().__new__(mcs, name, bases, dict_, **kwargs)
    dict_["__new__"] = new
    subcls = mcs(name, (cls,), dict_, _calledfromnew=True, **kwargs)
    return subcls

  __dataclassinits = {}
  @classmethod
  def dataclassinit(mcs):
    #if it's already DataClassSuperNewInit
    if issubclass(mcs, DataClassSuperNewInit): return mcs
    #if the class is DataClassMeta
    if issubclass(DataClassSuperNewInit, mcs): return DataClassSuperNewInit

    #make a DataClassSuperNewInit for the subclass of DataClassMeta
    if mcs not in mcs.__dataclassinits:
      class NewDataClassSuperNewInit(mcs, DataClassSuperNewInit):
        pass
      NewDataClassSuperNewInit.__name__ = mcs.__name__ + "Init"
      mcs.__dataclassinits[mcs] = NewDataClassSuperNewInit

    return mcs.__dataclassinits[mcs]

class DataClassSuperNewInit(DataClassSuperNewMeta, DataClassInit):
  def __new__(mcs, name, bases, dict_, **kwargs):
    dataclass_bases = [vars(b) for b in bases if hasattr(b, '__annotationmetadata__')]
    __noninitargs__ = set()
    for b in dataclass_bases + [dict_]:
      __noninitargs__ |= b.get("__noninitargs__", set())
    dict_["__noninitargs__"] = __noninitargs__
    return super().__new__(mcs, name, bases, dict_, **kwargs)

  """In the case that a custom __init__ is defined, remove arguments used by __new__ before calling it."""
  def __call__(cls, *args, **kwargs):
    args = iter(args)
    new_kwargs = dict(zip(cls.__annotations__, args))  # convert positional args to keyword for __new__
    instance = cls.__new__(cls, **new_kwargs, **kwargs)

    for parameter in kwargs.keys() & (cls.__annotations__.keys() | cls.__noninitargs__):
      del kwargs[parameter]

    instance.__init__(*args, **kwargs)
    return instance

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

class MyDataClassMeta(DataClassSuperNewMeta, DataClassWithMetaDataMeta, abc.ABCMeta):
  pass

@dataclassy.dataclass(meta=MyDataClassMeta)
class MyDataClass(DataClassWithMetaData):
  pass
