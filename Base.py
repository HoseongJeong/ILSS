import sys
import copy 
import re
from collections import defaultdict, deque
from bisect import bisect_right, bisect_left
from operator import attrgetter,eq,mul, truediv
import Module_wide_variable as MWV

__type__ = object

class Primitive(object):
    """Class that encapsulates a primitive and when called with arguments it
    returns the Python code to call the primitive with the arguments.

        >>> pr = Primitive("mul", (int, int), int)
        >>> pr.format(1, 2)
        'mul(1, 2)'
    """
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name, args, ret):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        args = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Terminal(object):
    """Class that encapsulates terminal primitive in expression. Terminals can
    be values or 0-arity functions.
    """
    __slots__ = ('name', 'value', 'ret', 'conv_fct')

    def __init__(self, terminal, symbolic, ret):
        self.ret = ret
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Ephemeral(Terminal):
    """Class that encapsulates a terminal which value is set when the
    object is created. To mutate the value, a new object has to be
    generated. This is an abstract base class. When subclassing, a
    staticmethod 'func' must be defined.
    """

    def __init__(self):
        Terminal.__init__(self, self.func(), symbolic=False, ret=self.ret)

    @staticmethod
    def func():
        """Return a random value used to define the ephemeral state.
        """
        raise NotImplementedError


class PrimitiveSetTyped(object):
    """Class that contains the primitives that can be used to solve a
    Strongly Typed GP problem. The set also defined the researched
    function return type, and input arguments type and number.
    """

    def __init__(self, name, in_types, ret_type, prefix="ARG"):
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        # setting "__builtins__" to None avoid the context
        # being polluted by builtins function when evaluating
        # GP expression.
        self.context = {"__builtins__": None}
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0

        self.name = name
        self.ret = ret_type
        self.ins = in_types
        for i, type_ in enumerate(in_types):
            arg_str = "{prefix}{index}".format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add(term)
            self.terms_count += 1

    def renameArguments(self, **kargs):
        """Rename function arguments with new names from *kargs*.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in list(dict_.items()):
                    if issubclass(type_, ret_type):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if issubclass(prim.ret, type_):
                dict_[type_].append(prim)

    def addPrimitive(self, primitive, in_types, ret_type, name=None):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :parma in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        """
        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)

        assert name not in self.context or \
               self.context[name] is primitive, \
            "Primitives are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second '%s' primitive." % (name,)

        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name=None):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        """
        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__

        assert name not in self.context, \
            "Terminals are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            # To support True and False terminals with Python 2.
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type)
        self._add(prim)
        self.terms_count += 1

    def addEphemeralConstant(self, name, ephemeral, ret_type):
        """Add an ephemeral constant to the set. An ephemeral constant
        is a no argument function that returns a random value. The value
        of the constant is constant for a Tree, but may differ from one
        Tree to another.

        :param name: name used to refers to this ephemeral type.
        :param ephemeral: function with no arguments returning a random value.
        :param ret_type: type of the object returned by *ephemeral*.
        """
        module_gp = globals()
        class_ = type(name, (Ephemeral,), {'func': staticmethod(ephemeral),
                                            'ret': ret_type})
        module_gp[name] = class_
        self._add(class_)
        self.terms_count += 1

    def addADF(self, adfset):
        """Add an Automatically Defined Function (ADF) to the set.

        :param adfset: PrimitiveSetTyped containing the primitives with which
                       the ADF can be built.
        """
        prim = Primitive(adfset.name, adfset.ins, adfset.ret)
        self._add(prim)
        self.prims_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)
        
        
class PrimitiveSet(PrimitiveSetTyped):
    """Class same as :class:`~deap.gp.PrimitiveSetTyped`, except there is no
    definition of type.
    """

    def __init__(self, name, arity, prefix="ARG"):
        args = [__type__] * arity
        PrimitiveSetTyped.__init__(self, name, args, __type__, prefix)

    def addPrimitive(self, primitive, arity, name=None):
        """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
        assert arity > 0, "arity should be >= 1"
        args = [__type__] * arity
        PrimitiveSetTyped.addPrimitive(self, primitive, args, __type__, name)

    def addTerminal(self, terminal, name=None):
        """Add a terminal to the set."""
        PrimitiveSetTyped.addTerminal(self, terminal, __type__, name)

    def addEphemeralConstant(self, name, ephemeral):
        """Add an ephemeral constant to the set."""
        PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, __type__)
        #Primitive tree
        
class PrimitiveTree(list):
    """Tree specifically formatted for optimization of genetic
    programming operations. The tree is represented with a
    list where the nodes are appended in a depth-first order.
    The nodes appended to the tree are required to
    have an attribute *arity* which defines the arity of the
    primitive. An arity of 0 is expected from terminals nodes.
    """
    
    def __init__(self, content):
        list.__init__(self, content)
    
    
    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                  " in a tree of size %d). Even if this is allowed by the"
                                  " list object slice setter, this should not be done in"
                                  " the PrimitiveTree context, as this may lead to an"
                                  " unpredictable behavior for searchSubtree or evaluate."
                                  % (key, len(self)))

            if "type" in str(type(val[0])): ##it was added for constant terminal
                total = 0
                for node in val[1:]:
                    total += node.arity - 1
            else:
                total = val[0].arity
                for node in val[1:]:
                    total += node.arity - 1

            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                  " an incomplete subtree is not allowed in PrimitiveTree."
                                  " A tree is defined as incomplete when some nodes cannot"
                                  " be mapped to any position in the tree, considering the"
                                  " primitives' arity. For instance, the tree [sub, 4, 5,"
                                  " 6] is incomplete if the arity of sub is 2, because it"
                                  " would produce an orphan node (the 6).")
        elif val.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                              " different arity.")
        list.__setitem__(self, key, val)
    
    def __str__(self):
        """Return the expression in a human readable string.
        """
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)
        return string
    
    @classmethod
    def from_string(cls, string, pset):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.
    
        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None
    
            if token in pset.mapping:
                primitive = pset.mapping[token]
    
                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))
    
                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))
    
                if type_ is None:
                    type_ = type(token)
    
                if not issubclass(type(token), type_):
                    raise TypeError("Terminal {} type {} does not "
                                    "match the expected one: {}."
                                    .format(token, type(token), type_))
    
                expr.append(Terminal(token, False, type_))
        return cls(expr)
    
    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth
    
    @property
    def root(self):
        """Root of the tree, the element 0 of the list.
        """
        return self[0]
    
    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    def searchChild(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        child=[]
        total = self[begin].arity
        begin_children=begin+1
        while total!=len(child):
            child.append(begin_children)
            end_children = begin_children+1
            total_children=self[begin_children].arity
            while total_children > 0:
                total_children += self[end_children].arity - 1
                end_children += 1
            begin_children=end_children
        return child
    
    def searchParent(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        parent=[]
        back_step=begin-1
        while begin not in self.searchChild(back_step):
            back_step-=1
        parent.append(back_step)
        return parent
    
    
    def searchPath(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        if len(self)!=1:
            path=[begin]
            while self.searchParent(begin)[0]!=0:
                path.append(self.searchParent(begin)[0])
                begin=path[-1]
            path.append(0)
            path.reverse()
            return path
        else :
            []
            
####################################################### HallOfFame #######################################################
class HallOfFame(object):
    """The hall of fame contains the best individual that ever lived in the
    population during the evolution. It is lexicographically sorted at all
    time so that the first element of the hall of fame is the individual that
    has the best first fitness value ever seen, according to the weights
    provided to the fitness at creation time.

    The insertion is made so that old individuals have priority on new
    individuals. A single copy of each individual is kept at all time, the
    equivalence between two individuals is made by the operator passed to the
    *similar* argument.

    :param maxsize: The maximum number of individual to keep in the hall of
                    fame.
    :param similar: An equivalence operator between two individuals, optional.
                    It defaults to operator :func:`operator.eq`.

    The class :class:`HallOfFame` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.
    """
    def __init__(self, maxsize, similar=eq):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar

    def update(self, population):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """     
        for ind in population:
            if len(self) == 0 and self.maxsize !=0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if ind.fitness < self[-1].fitness or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(-1)
                        self.insert(ind)

    def insert(self, item):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """
        item = copy.deepcopy(item)
        #높은게 좋은거면 이걸로
        # i = bisect_right(self.keys, item.fitness)
        # self.items.insert(len(self) - i, item)
        # self.keys.insert(i, item.fitness)
        #낮은게 좋은거면  이걸로
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.keys[index]
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]
        del self.keys[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    # def __reversed__(self):
    #     return reversed(self.items)

    def __str__(self):
        return str(self.items)    
        
    
def search_terminal(individual):
    index=[]
    for i in range(0,len(individual)):
        if isinstance(individual[i],Terminal):
            index.append(i)
    return index

def search_variable(individual):
    index_list=[]
    value_list=[]
    for i in range(0,len(individual)):
        if "constant" not in str(type(individual[i])):
            if isinstance(individual[i],Terminal):
                index_list.append(i)
                value_list.append(individual[i].value)
    return index_list,value_list

def search_primitive(individual):
    index=[]
    for i in range(0,len(individual)):
        if isinstance(individual[i],Primitive):
            if i!=0 or 1:
                index.append(i)
    return index

def search_constant(individual):
    index_list=[]
    value_list=[]
    for i in range(0,len(individual)):
        if "constant" in str(type(individual[i])):
            index_list.append(i)
            value_list.append(individual[i].value)
    return index_list,value_list
        
def linear_scailing(linear_scaling_tree,size_base_tree):
    tree1=copy.deepcopy(linear_scaling_tree)
    tree2=copy.deepcopy(size_base_tree)
    slice1=tree1.searchSubtree(3)
    slice2=tree2.searchSubtree(0)
    tree1[slice1]=tree2[slice2]
            
    return tree1