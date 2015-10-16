"""
Create, load and save datafiles in the Arts XML format.

For XML output, the main class is XMLfile.  An XMLfile object is
initialised with a filename or a stream.

>>> from PyARTS import artsXML
>>> testfile=artsXML.XMLfile('a_test_file.xml')

Arts data objects are then added to the file with the add method.

>>> a_tensor=ones([3,5,6],np.float32)
>>> testfile.add(a_tensor)

Context managers are also supported:

>>> with artsXML.XMLfile("a_test_file.xml") as tf:
...     tf.add(my_tensor)

Then you don't need to worry about closing the file, as the context
manager takes care of this.

The ARTS data type to save is is determined automatically from the python
type. This mapping is specified by the dictionary artsXML.MAPPING. For
instance, [[1,2,3],[4,5,6]] would be saved as an ArrayOfArrayOfIndex, whereas
[array([1,2,3]),array([4,5,6])] would be saved as an ArrayOfVector. Note that
it is not guaranteed that ARTS actually understands this! For Tensor type
objects, the tag name (eg. Tensor3) and the size attributes are determined
automatically by the shape of the np array.

The file must then be closed with the close() method:

>>> testfile.close()

A shortcut save function is available.  Using save the above
is achieved in one line.

>>> artsXML.save(a_tensor,'a_test_file.xml')

Some more complicated structures, like SingleScatteringData objects, have
their own save methods which utilize this module.

The load function is a low-level function that returns a dictionary structure
reflecting the structure of the XML file. If you already know the type of the
data you are going to load, you can use the particular constructor, such as
SingleScatteringData.load or ArrayOfLatLonGriddedField3.load. All of these
will use artsXML.load as a backend.
"""

from __future__ import print_function

import numpy as np
import copy

from io import StringIO

from io import BytesIO

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)

import contextlib
import gzip

from . import general

import xml.sax
import xml.sax.saxutils

from collections import OrderedDict

##CONSTANTS

ARTS_DIM_LABELS = ["ncols", "nrows", "npages", "nbooks",
                   "nshelves", "nvitrines", "nlibraries"]

ARTS_TENSOR_NAMES = ["Vector", "Matrix", "Tensor3", "Tensor4",
                     "Tensor5", "Tensor6", "Tensor7"]

ARTS_TEXT_NAMES = ["String", "comment", "SpeciesTag"]

ARTS_SPARSE_NAMES = ["RowIndex", "ColIndex", "SparseData"]

# controls the verbosity of all classes and functions in this module
VERBOSITY = False


##CLASSES

class Sparse(object):
    """Sparse object

    Attributes attributes are gotten via XML parsing
    (nrows, ncols, etc.)
    """

    def __init__(self, attributes):
        if VERBOSITY:
            print("creating Sparse matrix")
        dimlist = []
        dimlist.append(int(attributes.getValue(ARTS_DIM_LABELS[1])))
        dimlist.append(int(attributes.getValue(ARTS_DIM_LABELS[0])))
        if VERBOSITY:
            print("dimensions:")
            print(dimlist)
        self.dimlist = dimlist
        self.sparse = {}

    def setdata(self, tag, v):
        if tag in ARTS_SPARSE_NAMES:
            self.sparse[tag] = v
        else:
            raise TypeError("Unknown member in Sparse type: %s" % tag)

    def csc_matrix(self):
        """ Returns a scipy sparse object """
        import scipy.sparse

        return scipy.sparse.csc_matrix(
            (self.sparse['SparseData'],
             (self.sparse['RowIndex'], self.sparse['ColIndex'])),
            self.dimlist)


class Tensor(object):
    """Tensor object

    Rank, attributes attributes are gotten via XML parsing
    (nelem, nlibraries, etc.)
    """

    def __init__(self, rank, attributes):
        if VERBOSITY:
            print("creating Tensor of rank {0}".format(rank))
        dimlist = []
        if rank == 1:
            dimlist.append(int(attributes.getValue("nelem")))
        else:
            for i in range(rank):
                dimlist.append(int(attributes.getValue(ARTS_DIM_LABELS[i])))
        # put the highest dimensions first
        dimlist.reverse()
        if VERBOSITY:
            print("dimensions:")
            print(dimlist)
        # fill with nans
        Z = np.zeros(dimlist)
        Z.fill(np.nan)
        self.array = Z
        self.dimlist = dimlist

    def setdata(self, v):
        if v.size != np.prod(self.dimlist):
            raise TypeError("Shape mismatch: data %s, dimlist %s"
                            % (v.shape, self.dimlist))
        self.array = v
        self.fixshape()

    def fixshape(self):
        if VERBOSITY:
            print("raw data shape = ")
            print(self.array.shape)
        self.array.shape = self.dimlist


# Mapping between Python and ARTS types
MAPPING = {
    list: 'Array',
    int: 'Index',
    float: 'Numeric',
    str: 'String',
    np.ndarray: 'Tensor',
    Sparse: 'Sparse',
}


class XML_Obj(object):
    def __init__(self, tag, attributes=None):
        if attributes is None:
            attributes = {}

        self.tag = tag
        self.s = BytesIO()
        self.write(
            (u'<' + tag + _attributeconvert(attributes) + '>\n'))

    def write(self, content):
        self.s.write(general.convert_to_bytes(content))
        return self

    def finalise(self):
        self.write(('</' + self.tag + '>\n'))
        self.str = self.s.getvalue()
        self.s.close()
        return self


class XMLfile(object):
    """Arts XML output class.

    Initialise with a filename or object.
    Warning: this will open the file for writing, thus over-writing any
    existing data!
    """

    def __init__(self, f, header=True):
        if isinstance(f, basestring):
            self.filename = f
            self.file = open(f, 'wb')
        else:
            self.file = f
        if header:
            self.file.write('<?xml version="1.0"?>\n'.encode())
            self.file.write('<arts format="ascii" version="1">\n'.encode())
        self.header = header

    def _addnumber(self, tag, number, attributes={}):
        self.file.write(
            general.convert_to_bytes(number_to_xml(tag, number, attributes)))

    def _addText(self, tag, text, attributes={}):
        self.file.write(
            general.convert_to_bytes(text_to_xml(tag, text, attributes)))

    def _addTensor(self, tensor):
        """
        This method takes a np array argument and stores it in
        the arts XML format with the appropriate Tag
        (e.g. Vector, Matrix, ...)
        """
        tensor_to_xml(tensor, self.file)

    def add(self, data):
        """
        Determine the ARTS type from the python type:
        Python list => ARTS array,
        np.ndarray => ARTS Vector/Tensor,
        Python string => ARTS string,
        Python integer => ARTS Index,
        Python float => ARTS Numeric
        """
        if 'np.int' in str(type(data)):
            self._addnumber('Index', data)

        # Python list => ARTS array
        elif MAPPING[type(data)] == 'Array':
            self.file.write(('<Array type="%s" nelem="%d">\n'
                             % (get_arts_type(data[0]), len(data))).encode())
            for x in data:
                self.add(x)
            self.file.write('</Array>\n'.encode())
        elif MAPPING[type(data)] == 'Tensor':
            self._addTensor(data)
        elif MAPPING[type(data)] == 'String':
            self._addString(data)
        elif MAPPING[type(data)] in ['Numeric', 'Index']:
            self._addnumber(MAPPING[type(data)], data)

    def _addString(self, data):
        """adds a String"""
        self.file.write('<String>"'.encode())
        self.file.write(xml.sax.saxutils.escape(data).encode())
        self.file.write('"</String>\n'.encode())

    def close(self, really=True):
        """This must be called to finalise the XML file"""
        if self.header:
            self.file.write(general.convert_to_bytes('</arts>\n'))
        if really:
            self.file.close()

    def write(self, s):
        self.file.write(general.convert_to_bytes(s))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Handler(xml.sax.ContentHandler):
    """
    XML handler for Arts files.

    This is much more general then the old one. Unlike the previous handler,
    this handler works by continutally updating a list of strings that
    constitute a reference for the eventual location of an object within
    the overall data structure. With the exception of Arrays, which are
    represented by lists, every XML tag becomes a dictionary key. In cases
    where the tag line has a name attribute, the dictionary key becomes the
    name string. This behaviour is enabled by default but can be controlled
    with the une_names argument on initialisation.

    Normally it should not be necessary to use this function directly, instead
    use the newLoad function below.
    """

    def __init__(self, use_names=True):
        """
        If use_names is True, the handler will use any present "name"
        attributes as keys for each object. Otherwise tags are used
        """
        self.CurrentHierarchy = []
        self.datastruct = OrderedDict()
        self.reference_str_list = ["self.datastruct"]
        # this is used to ensure nothing is overwritten
        self.assignment_str_list = []
        self.use_names = use_names

    def characters(self, text):
        try:
            self.buff.write(text)
        except ValueError:
            pass  # for historical reasons

    def startDocument(self):
        if VERBOSITY:
            print("Starting to parse XML")

    def startElement(self, tag, attributes):
        """Called when a new element starts"""

        def advance_name_if_nec(name, reference_str_list):
            if (eval('"' + name + '"'
                         + " in " + "".join(self.reference_str_list))):
                # then change the name
                if VERBOSITY:
                    print('already have ' + name)
                if (len(name.split()) == 1):
                    name = name + ' 0'
                else:
                    label = name.split()[0]
                    number = int(name.split()[1])
                    name = label + ' ' + str(number + 1)
                if VERBOSITY:
                    print("name changed to " + name)
                name = advance_name_if_nec(name, reference_str_list)
            return name

        self.buff = StringIO()
        if self.use_names:
            name = attributes.get("name", tag)
        else:
            name = tag
        if VERBOSITY:
            print("begin Name: " + name + "  Tag: " + tag)
        # Use tag to modify reference string for subsequent data
        if tag == "Array":
            name = advance_name_if_nec(name, self.reference_str_list)
            self.CurrentHierarchy.append(name)

            # [{}]*n gives n references to the same dict!!!!
            list_of_empty_dicts = []
            for i in range(int(attributes["nelem"])):
                list_of_empty_dicts.append(OrderedDict())
            eval("".join(self.reference_str_list)
                 + '.update({"' + name + '":list_of_empty_dicts})'
                 )
            self.reference_str_list.append('["' + name + '"]')
            self.reference_str_list.append("[0]")
            self.CurrentHierarchy.append(0)
        else:
            # check that new object doesn't already exist

            name = advance_name_if_nec(name, self.reference_str_list)

            eval("".join(self.reference_str_list)
                 + '.update({"' + name + '":OrderedDict()})')
            self.reference_str_list.append('["' + name + '"]')
            self.CurrentHierarchy.append(name)

            # If appropriate, create a tensor object
        if tag in ARTS_TENSOR_NAMES:
            rank = ARTS_TENSOR_NAMES.index(tag) + 1
            self.CurrentObject = Tensor(rank, attributes)
        elif tag == 'Sparse':
            self.CurrentObject = Sparse(attributes)

    def text2obj(self, tag, text):
        """Converts XML-text to appropiate object"""

        text = text.strip()
        if tag in ARTS_TENSOR_NAMES:
            self.CurrentObject.setdata(np.loadtxt(BytesIO(text.encode())))
            # self.CurrentObject.fixshape()
            return self.CurrentObject.array
        elif tag in ARTS_SPARSE_NAMES:
            if tag[3:] == 'Index':
                self.CurrentObject.setdata(tag, np.loadtxt(StringIO(text),
                                                           dtype=np.int32))
            else:
                self.CurrentObject.setdata(tag, np.loadtxt(StringIO(text)))
            return self.CurrentObject.sparse[tag]
        elif tag in ARTS_TEXT_NAMES:
            return text
        elif tag == "Index":
            return np.int64(text)
        elif tag == "Numeric":
            return np.float64(text)
        elif tag == "Sparse":
            return self.CurrentObject
        else:
            if text.strip():
                raise TypeError("Cannot convert to obj: %s" % tag)
            else:
                return None

    def endElement(self, tag):
        if not self.buff.closed:
            text = self.buff.getvalue()
            self.buff.close()
        else:
            text = ""
        if VERBOSITY:
            print("end  Tag: " + tag)
        obj = self.text2obj(tag, text)
        if obj is not None:
            assignment_str = ("".join(self.reference_str_list[:-1])
                              + '.update({"'
                              + self.CurrentHierarchy[-1]
                              + '":obj})')
            if VERBOSITY:
                print(assignment_str)
            eval(assignment_str)
        elif VERBOSITY:
            print("No text?")

        self.CurrentHierarchy.pop()
        self.reference_str_list.pop()
        if tag == "Array":
            self.CurrentHierarchy.pop()
            self.reference_str_list.pop()
        # if necessary advance array index
        if self.CurrentHierarchy:
            ref = self.CurrentHierarchy[-1]
            if isinstance(ref, int):
                ref += 1
                self.CurrentHierarchy[-1] = ref
                self.reference_str_list[
                    len(self.CurrentHierarchy)] = "[" + str(ref) + "]"


##FUNCTIONS

def _attributeconvert(attributes):
    attributestr = ''
    for key in attributes.keys():
        attributestr += ' ' + str(key) + '="' + str(attributes[key]) + '"'
    return attributestr


def get_arts_type(data):
    """Returns the equivalent ARTS data type for a given Python object"""
    # nasty catch for np integers, which aren't behaving as expected
    if isinstance(data, np.integer):
        return 'Index'
    elif MAPPING[type(data)] == 'Array':
        return 'ArrayOf' + get_arts_type(data[0])
    elif MAPPING[type(data)] == 'Tensor':
        return ARTS_TENSOR_NAMES[data.ndim - 1]
    else:
        return MAPPING[type(data)]


def number_to_xml(tag, number, attributes={}):
    return XML_Obj(tag, attributes).write(repr(number) + '\n').finalise().str


def text_to_xml(tag, text, attributes={}):
    return XML_Obj(tag, attributes).write(xml.sax.saxutils.escape(text)
                                          + '\n').finalise().str


def tensor_to_xml(tensor, outfile=None):
    """Store tensor in XML.

    This method takes a Python array  argument and stores it in
    the arts XML format with the appropriate Tag (eg Vector Matrix,
    if outfile (a file object - not a name) is given then it is written to
    that file
    """
    return_string = False
    oldshape = copy.deepcopy(tensor.shape)
    rank = tensor.ndim
    dim_labels = ARTS_DIM_LABELS[:rank]
    dim_labels.reverse()
    if (rank == 1):
        dim_labels[0] = "nelem"
    tag = ARTS_TENSOR_NAMES[rank - 1]
    attributes = {}
    for i in range(rank):
        attributes.update({dim_labels[i]: repr(tensor.shape[i])})
    if (rank > 2):
        # reshape tensor to rank two for output
        tensor = tensor.ravel().reshape(-1, tensor.shape[-1])

    if outfile is None:
        return_string = True
        outfile = XML_Obj(tag, attributes)
        fp = outfile.s
    else:
        outfile.write(
            ('<' + tag + _attributeconvert(attributes) + '>\n').encode())
        fp = outfile

    # use only %.7e, good enough for float
    np.savetxt(fp, tensor, delimiter=" ", fmt="%.7e")
    tensor.reshape(oldshape)
    if return_string:
        return outfile.finalise().str
    else:
        outfile.write(('</' + tag + '>\n').encode())


def save(data, filename):
    """Saves data to arts XML file (filename or stream).

    If the filename ends in '.gz' the XML file will be gzipped
    """
    try:
        data.save(filename)
    except AttributeError:
        if filename.lower().endswith(".gz"):
            opener = contextlib.closing(gzip.GzipFile(filename, "w"))
        else:
            opener = open(filename, "wb")
        with opener as fp:
            outfile = XMLfile(fp)
            outfile.add(data)
            outfile.close()


def load(filename, use_names=True):
    """Loads an ArtsXML-file.

    This general purpose function returns a dictionary structure reflecting
    the structure of the XML file. If there is only one object in the
    structure, then that single object is returned. As far as far as I know,
    this works with every data type exported by ARTS. Note that only in
    special simple cases, such as single tensors, will load return exactly the
    same data given to the save command. The more complicated data structures
    (e.g. SingleScatteringData) have their own load methods. gzipped files can
    be loaded (as long as they end in 'gz').

    Parameters
    ~~~~~~~~~~

    filename : string-like
        Path to file to load (.xml or .xml.gz)

    use_names : boolean, optional (default: True)
        Use name tags in XML-file

    Returns
    ~~~~~~~

    Returns an OrderedDict with the contents of the XML-file.
    """
    if filename.lower().endswith(".gz"):
        opener = contextlib.closing(gzip.GzipFile(filename, "r"))
    else:
        opener = open(filename, "rb")
    handler = Handler(use_names)
    with opener as fp:
        xml.sax.parse(fp, handler)
    data = handler.datastruct["arts"]
    if len(data) == 1:
        return list(data.values())[0]
    else:
        return data
