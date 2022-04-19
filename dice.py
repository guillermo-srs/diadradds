import numpy as np
from numpy.random import permutation as rperm
from itertools import product as prod, zip_longest as zl
import seaborn as sb
import matplotlib.pyplot as plt


# Dice combination functions

def poker_value(lst):
    """Function that combines a list of values in sets of consecutive and
    repetitive elements, and then returns biggest sum.

    Parameters
    ----------
    lst : list
       List of elements to group

    Returns
    -------
    int
       Maximum value between the different subsets (of consecutive or
    repetitive elements) within lst
    """

    nums, rep = np.unique(lst, return_counts=True)
    # The maximum value between repetitions and consecutive, elements
    # is returned
    return max(
        # For each sub list of consecutive elements its sum is measured
        # and the maximum value is taken
        max(map(lambda x: np.sum(x), get_consecutives(lst))),
        # Each element is multiplied for its number of apparitions in lst
        np.max(nums*rep))


# This 4 methods are defined to encapsulate functions and unify nomenclature
def median_value(lst):
    return np.median(lst)


def sum_value(lst):
    return np.sum(lst)


def max_value(lst):
    return max(lst)


def min_value(lst):
    return min(lst)


def grouper(iterable, n, fillvalue=None):
    """Function obtained from https://docs.python.org/2/library/itertools.html
    to group elements in the array 'iterable' in subsets of 'n' elements

    Parameters
    ----------
    iterable : array-like
       Array of elements to group
    n : int
       Size of the groups

    Returns
    -------
    Array-like
       2D array of the elements of iterable grouped in groups of size n
    """
    args = [iter(iterable)] * n
    return zl(*args, fillvalue=fillvalue)


def dice_distribution(num_dices, f=poker_value, value=10,
                      method="simple", fctr=0.1, mn=2000):
    """Function to estimate the probability distribution of throw 'num_dices'
    dices of 'value' faces (stating on 0), and combining them
    with a 'f' function

    Parameters
    ----------
    num_dices : int
        Number of dices
    f : function (default=poker_value)
        Function used to obtain the value of each throw
    value : int (default=10)
        Number of faces of each dice
    method : str (default="simple")
        Method used in the estimation of the probability distribution

           - simple:   The estimation is perform over a subset of 'mn'
                       elements grouped in groups of size 'num_dices'
           - half:     The total number of throw combinations is calculated
                       and then a subset of size 'mn' is taken. This is method
                       is not recommended because it has high computational
                       cost with num_dices > 5.
           - complete: No estimation is perform in this case. The total number
                       of combinations is used instead. This is method
                       is not recommended because it has high computational
                       cost with num_dices > 3.
    fctr : float
       reduction factor used to reduce the number of subsets used in the
       complete method. With any other method, this parameter is ignored.
    mn : int
       Number of throws used for the methods 'half' and 'simple'. If the
       method 'complete' is selected, it uses both 'mn' and 'fctr'.

    Returns
    -------
    list
       List with the values obtained from the simulation
    """

    if method == "simple":
        # A random int array with shape multiple num_dices is generated
        f_arr = np.random.randint(0, value, int(mn/num_dices)*num_dices)
        # then the array is grouped in sub lists of num_dices elements
        # and evaluated
        return list(map(lambda x: f(x), grouper(f_arr, num_dices)))

    f_arr = [np.arange(value) for i in range(num_dices)]
    # list of lists of possible combinations of each dice
    if method == "half":
        # for the 'half' method a sub sample of combinations is calculated
        aux = np.array(list(prod(*f_arr)))
        lst = aux[rperm(len(aux))][:int(max(mn, len(aux)*fctr))]
    elif method == "full":
        lst = prod(*f_arr)
    else:
        return None

    return list(map(lambda x: f(x), lst))


def get_consecutives(lst):
    """Function that receives a list of integers and returns a list with each sub
    list of consecutive numbers.

    Parameters
    ----------

    lst : array-like
        list of integers

    Returns
    -------
    list of lists
        List composed of list of consecutive numbers contained in lst

    """
    return get_consecutives_rec(np.unique(np.sort(lst)).tolist())


def get_consecutives_rec(lst):
    """Auxiliary function of get_consecutives to perform recursion

    Parameters
    ----------

    lst : array-like
        list of sorted integers without repetitions

    Returns
    -------
    list of lists
        List composed of list of consecutive numbers contained in lst

    """
    if len(lst) == 1:
        # recursion end
        return [lst]
    aux_lst = get_consecutives_rec(lst[1:])
    if aux_lst[0][0] - 1 == lst[0]:
        aux_lst[0].insert(0, lst[0])
    else:
        aux_lst.insert(0, [lst[0]])
    return aux_lst


# Plotting functions

def plot_dist(dst, bns=-1, name="", diff_lim=0.94):

    sb.set()
    mn_dst = np.min(dst)
    mx_dst = np.max(dst)
    if bns == -1:
        bns = int(mx_dst-mn_dst)

    plt.figure(figsize=[7.5, 6.5])
    plt.subplot(211)
    plt.title("Distribution: " + name)
    xs = sb.distplot(dst, bins=bns)
    bin_edges = np.histogram(dst, bins=8, density=True)[1]

    sb.distplot(dst, bins=8, label='x'+str(bin_edges[2]-bin_edges[1])[:3])
    plt.xticks(np.around(np.linspace(mn_dst-1, mx_dst+1, bns+2)), rotation=45)
    plt.xlim(mn_dst-1, mx_dst+1)
    plt.legend()

    xl = xs.get_xlim()
    yl = xs.get_ylim()
    plt.yticks(np.around(np.linspace(0., yl[1], 8), decimals=3))

    def adjst(x):
        xs = x[1] - x[0]
        return xs - 0.15*xs

    plt.text(adjst(xl), adjst(yl)-1*yl[1]/10, 'Mean: '+str(np.mean(dst))[:3])
    plt.text(adjst(xl), adjst(yl)-2*yl[1]/10,
             'Median: ' + str(np.median(dst))[:3])
    plt.text(adjst(xl), adjst(yl)-3*yl[1]/10, 'Max: '+str(mx_dst)[:3])
    plt.text(adjst(xl), adjst(yl)-4*yl[1]/10, 'Min: '+str(mn_dst)[:3])

    plt.subplot(223)
    sb.distplot(dst, kde_kws={'cumulative': True})
    hist, bin_edges = np.histogram(dst, density=True)
    cumu_hist = np.cumsum(hist*(bin_edges[1] - bin_edges[0]))
    m1 = np.where(cumu_hist > diff_lim)[0][0]
    m2 = m1 + 1

    plt.ylim(0., cumu_hist[m1])
    plt.yticks(np.around(np.linspace(0., cumu_hist[m1], 11), decimals=1))
    plt.xlim(mn_dst-1, bin_edges[m1])
    plt.xticks(np.around(
        np.linspace(mn_dst-1, bin_edges[m1], 14)), rotation=45)

    plt.subplot(224)
    sb.distplot(dst, hist_kws={'cumulative': True})
    plt.ylim(cumu_hist[m2], 1.)
    plt.yticks(np.around(
        np.linspace(cumu_hist[m2], 1., 10), decimals=3))
    plt.xlim(bin_edges[m2], mx_dst)
    plt.xticks(np.around(
        np.linspace(bin_edges[m2], mx_dst+1, 10)), rotation=45)
    plt.tight_layout()
    plt.savefig(name+'.png', format='png', dpi=600)


def demo(num_dices, value, f=poker_value, plot=False):
    """ Demo to test the capabilities of the library
    """
    dst = dice_distribution(num_dices, value=value, f=f)
    name = "F" + str(f) + "-" + str(num_dices) + "D" + str(value)
    np.save('data_'+name, dst)
    if plot is True:
        plot_dist(dst)


if __name__ == '__main__':
    demo(3, 10, plot=True)
