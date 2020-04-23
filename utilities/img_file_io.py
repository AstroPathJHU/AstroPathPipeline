import numpy as np

#helper function to read the binary dump of a raw im3 file 
def im3readraw(f) :
  with open(f,mode='rb') as fp : #read as binary
    content = np.fromfile(fp,dtype=np.uint16)
  return content

#helper function to write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
  with open(outname,mode='wb') as fp : #write as binary
    a.tofile(fp)

#helper function to read a raw image file and return it as an array of shape (height,width,n_layers)
def getRawAsHWL(fname,height,width,nlayers) :
  #get the .raw file as a vector of uint16s
  img = im3readraw(fname)
  #reshape it to the given dimensions
  try :
    img_a = np.reshape(img,(nlayers,width,height),order="F")
  except ValueError :
    msg = f"ERROR: Raw image file shape ({nlayers} layers, {len(img)} total bytes) is mismatched to"
    msg+= f" dimensions (layers, width={width}, height={height})!"
    raise ValueError(msg)
  #flip x and y dimensions to display image correctly, move layers to z-axis
  img_to_return = np.transpose(img_a,(2,1,0))
  return img_to_return

#helper function to read a single-layer image and return it as an array of shape (height,width)
def getRawAsHW(fname,height,width) :
  #get the file as a vector of uint16s
  img = im3readraw(fname)
  #reshape it
  try :
    img_a = np.reshape(img,(height,width),order="F")
  except ValueError :
    msg = f"ERROR: single layer image file ({len(img)} total bytes) shape is mismatched to"
    msg+= f" dimensions (width={width}, height={height})!"
    raise ValueError(msg)
  return img_a

#helper function to flatten and write out a given image as binary uint16 content
def writeImageToFile(img_array,filename_to_write) :
  #write out image flattened in fortran order
  im3writeraw(filename_to_write,img_array.flatten(order="F").astype(np.uint16))