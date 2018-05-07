classdef PrintDMatrix < SwigRef
    %Base class for objects that have a natural string representation.
    %
    %Joel Andersson
    %
    %C++ includes: printable_object.hpp 
    %Usage: PrintDMatrix ()
    %
  methods
    function this = swig_this(self)
      this = casadiMEX(3, self);
    end
    function varargout = getDescription(self,varargin)
    %Return a string with a description (for SWIG)
    %
    %
    %Usage: retval = getDescription ()
    %
    %retval is of type std::string. 

      try

      if ~isa(self,'casadi.PrintDMatrix')
        self = casadi.PrintDMatrix(self);
      end
      [varargout{1:max(1,nargout)}] = casadiMEX(40, self, varargin{:});

      catch err
        if (strcmp(err.identifier,'SWIG:OverloadError'))
          msg = [swig_typename_convertor_cpp2matlab(err.message) 'You have: ' strjoin(cellfun(@swig_typename_convertor_matlab2cpp,varargin,'UniformOutput',false),', ')];
          throwAsCaller(MException(err.identifier,msg));
        else
          rethrow(err);
        end
      end

    end
    function varargout = getRepresentation(self,varargin)
    %Return a string with a representation (for SWIG)
    %
    %
    %Usage: retval = getRepresentation ()
    %
    %retval is of type std::string. 

      try

      if ~isa(self,'casadi.PrintDMatrix')
        self = casadi.PrintDMatrix(self);
      end
      [varargout{1:max(1,nargout)}] = casadiMEX(41, self, varargin{:});

      catch err
        if (strcmp(err.identifier,'SWIG:OverloadError'))
          msg = [swig_typename_convertor_cpp2matlab(err.message) 'You have: ' strjoin(cellfun(@swig_typename_convertor_matlab2cpp,varargin,'UniformOutput',false),', ')];
          throwAsCaller(MException(err.identifier,msg));
        else
          rethrow(err);
        end
      end

    end
    function self = PrintDMatrix(varargin)
      if nargin==1 && strcmp(class(varargin{1}),'SwigRef')
        if varargin{1}~=SwigRef.Null
          self.swigPtr = varargin{1}.swigPtr;
        end
      else

      try

        tmp = casadiMEX(42, varargin{:});
        self.swigPtr = tmp.swigPtr;
        tmp.swigPtr = [];

      catch err
        if (strcmp(err.identifier,'SWIG:OverloadError'))
          msg = [swig_typename_convertor_cpp2matlab(err.message) 'You have: ' strjoin(cellfun(@swig_typename_convertor_matlab2cpp,varargin,'UniformOutput',false),', ')];
          throwAsCaller(MException(err.identifier,msg));
        else
          rethrow(err);
        end
      end

      end
    end
    function delete(self)
      if self.swigPtr
        casadiMEX(43, self);
        self.swigPtr=[];
      end
    end
  end
  methods(Static)
  end
end
