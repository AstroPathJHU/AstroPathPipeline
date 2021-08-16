"""
The dataclasses in the astropath framework use the dataclassy package,
which has several features that make it easier to use than the built-in
dataclasses module.

Here we add some subclasses of the default dataclassy metaclass
to add some more features that will be useful for us.
"""

import abc
from dataclassy import dataclass
from dataclassy.dataclass import DataClassMeta

class DataClassTransformArgsMeta(DataClassMeta):
  """
  Metaclass for a dataclass that can take arguments that are
  not exactly the dataclass fields.  See DataClassTransformArgs
  for further help.
  """
  def __call__(cls, *args, **kwargs):
    newargs, newkwargs = cls.transforminitargs(*args, **kwargs)
    return super().__call__(*newargs, **newkwargs)

class DataClassTransformArgs(metaclass=DataClassTransformArgsMeta):
  """
  Dataclass that can take arguments that are not exactly the dataclass fields
  and pass calculated values to the dataclass fields.
  For example:

    >>> class Vector(DataClassTransformArgs):
    ...  x: float
    ...  y: float
    ...  @classmethod
    ...  def transforminitargs(cls, *args, xvec=None, **kwargs):
    ...    morekwargs = {}
    ...    if xvec is not None:
    ...      morekwargs["x"], morekwargs["y"] = xvec
    ...    return super().transforminitargs(*args, **kwargs, **morekwargs)

    >>> Vector(1, 2)
    Vector(x=1, y=2)
    >>> Vector(x=1, y=2)
    Vector(x=1, y=2)
    >>> Vector(xvec=[1, 2])
    Vector(x=1, y=2)

  """
  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    return args, kwargs

class DataClassWithMetaDataMeta(DataClassMeta):
  """
  Metaclass for a dataclass that can have metadata, similar to
  the functionality in python's dataclasses module.  See
  DataClassWithMetaData for more help.
  """
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

class DataClassWithMetaData(metaclass=DataClassWithMetaDataMeta):
  """
  Dataclass that can have metadata, similar to the functionality
  in python's dataclasses module.  To create a field with an
  annotation, set a MetaDataAnnotation as the "default" value
  of a field.  Any keyword arguments to the MetaDataAnnotation
  will be the metadata for the field.  If you give it a positional
  argument, that will be the default, otherwise there is no default.

  You can then access the metadata using the metadata function.

  For example:

    >>> class MetaDataDataClass(DataClassWithMetaData):
    ...  field1: int
    ...  field2: int = MetaDataAnnotation(myname="field2")
    ...  field3: int = 3
    ...  field4: int = MetaDataAnnotation(4, myname="field4")

    >>> a = MetaDataDataClass(1, 2)
    >>> a
    MetaDataDataClass(field1=1, field2=2, field3=3, field4=4)
    >>> a.metadata("field2")["myname"]
    'field2'
    >>> a.metadata("field4")["myname"]
    'field4'
    >>> a.metadata("field3")["myname"]
    Traceback (most recent call last):
      ...
    KeyError: 'myname'
    >>> MetaDataDataClass()
    Traceback (most recent call last):
      ...
    TypeError: __init__() missing 2 required positional arguments: 'field1' and 'field2'

  """
  @classmethod
  def metadata(cls, fieldname):
    """
    Get the metadata for fieldname as a dict
    """
    return cls.__annotationmetadata__.get(fieldname, {})

class MetaDataAnnotation:
  """
  Set this as a default value of a field for a DataClassWithMetaData
  to assign metadata to a field.
  """
  __notgiven = object()
  def __init__(self, __defaultvalue=__notgiven, **kwargs):
    if __defaultvalue is not self.__notgiven:
      self.defaultvalue = __defaultvalue
    self.metadata = kwargs

class MyDataClassMeta(abc.ABCMeta, DataClassTransformArgsMeta, DataClassWithMetaDataMeta):
  """
  Metaclass for a dataclass that has both transforminitargs and metadata.
  It also inherits from ABCMeta so that you can use those features.
  See the parent classes' help messages for more information.
  """

class MyDataClass(DataClassTransformArgs, DataClassWithMetaData, metaclass=MyDataClassMeta):
  """
  Dataclass that has both transforminitargs and metadata.
  Its metaclass inherits from ABCMeta so that you can use those features.
  See the parent classes' help messages for more information.

  Also, this class has a built-in __post_init__ method with a readingfromfile
  argument.  This argument is ignored, but is passed by the readtable function
  in tableio.py and used in the units module.

  All the dataclasses used in astropath that read from a file should inherit
  from MyDataClass.
  """
  def __post_init__(self, *, readingfromfile=False, extrakwargs={}):
    pass

@dataclass(meta=MyDataClassMeta, frozen=True)
class MyDataClassFrozen(MyDataClass):
  """
  Frozen version of MyDataClass
  """
@dataclass(meta=MyDataClassMeta, unsafe_hash=True)
class MyDataClassUnsafeHash(MyDataClass):
  """
  Version of MyDataClass with the unsafe hash enabled
  """
