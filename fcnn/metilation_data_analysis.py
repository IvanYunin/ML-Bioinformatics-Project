import argparse
import matplotlib.pyplot as plot
from collections import Counter
import math

def load_data(metilation):
    file = open(metilation, 'r')
    header = file.readline()
    metilation_table_attributes = header.split('\t')
    metilation_table_attributes.pop(0)
    cpg_sites = []
    metilation_values = [ [] for i in range(len(metilation_table_attributes)) ]
    for line in file:
        values = line.split('\t')
        print('\t CpG-site: {}'.format(values[0]))
        cpg_sites.append(values.pop(0))

        for i in range(len(values)):
            metilation_values[i].append(float(values[i]))
    file.close()
    return [metilation_values, cpg_sites]


def load_attributes(attributes):
    file = open(attributes, 'r')
    file.readline()
    ages = []
    for line in file:
        values = line.split(' ')
        ages.append(int(values[2])) # age column
    file.close()
    return ages


def hist_class_samples(person_ages, quiet = False,
        hist_filename = 'hist_class_samples.pdf'):
    histogram_ages=dict(sorted(Counter(person_ages).items()))
    plot.bar(histogram_ages.keys(), histogram_ages.values(),
        color = 'steelblue')
    plot.xlabel('Age, year')
    plot.ylabel('Number of patients')
    if quiet:
        plot.savefig(hist_filename)
    else:
        plot.show()
    plot.gcf().clear()


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def calculate_mean_stds(metilation_values, person_ages,
        clofinterest):
    num_cpg_sites = len(metilation_values[0])
    
    indices = duplicates(person_ages, clofinterest)
    print('{0}: {1}'.format(clofinterest, indices))
    n = len(indices)
    if n == 0:
        raise ValueError('There are no patients of the specified age {}'.
            format(clofinterest))

    means = [ 0 for i in range(num_cpg_sites) ]
    for i in indices:
        means = [ means[idx] + metilation_values[i][idx] for idx in range(num_cpg_sites) ]
    means = [ means[i] / n for i in range(num_cpg_sites) ]

    stds = [ 0 for i in range(num_cpg_sites) ]
    for i in indices:
        stds = [ (a - ma) * (a - ma) for a, ma in zip(metilation_values[i], means) ]
    if n != 1:
        stds = [ math.sqrt(stds[i] / (n - 1)) for i in range(num_cpg_sites) ]
    else:
        stds = [ 0 for i in range(num_cpg_sites) ]
    
    return [means, stds, n]

def hist_mean_std_age(metilation_values, person_ages, clofinterest, quiet = False,
        hist_mean_std = 'hist_mean_std_{}.pdf'):
    [means, stds, n] = calculate_mean_stds(metilation_values, person_ages,
        clofinterest)
    x = [ i for i in range(len(means)) ]
    mean_hist = plot.bar(x, means, color = 'lightsteelblue')
    plot.xticks(x, [ str(i) for i in range(len(means))] )
    
    x = [ i for i in range(len(stds)) ]
    std_hist = plot.(x, stds, color = 'steelblue')
    plot.xticks(x, [ str(i) for i in range(len(stds))] )
    plot.xlabel('CpG-site index')
    plot.ylabel('Metilation value')
    plot.legend([mean_hist, std_hist], ['Mean value', 'Standard deviation'])
    if quiet:
        plot.savefig(hist_mean_std.format(clofinterest))
    else:
        plot.show()
    plot.gcf().clear()


def hist_mean_std(metilation_values, person_ages, clofinterest, quiet = False,
        hist_mean_std = 'hist_mean_std_{}.pdf'):
    if (clofinterest not in person_ages) and (clofinterest != 0):
        raise ValueError('Unsupported class identifier {}'.format(clofinterest))
    if clofinterest in person_ages:
        hist_mean_std_age(metilation_values, person_ages, clofinterest, quiet,
            hist_mean_std)
    else:
        all_ages = Counter(person_ages).keys()
        for clofinterest in all_ages:
            hist_mean_std_age(metilation_values, person_ages, clofinterest,
                quiet, hist_mean_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--metilation', help = 'metilation file name',
        default = 'GSE40279_average_beta.txt')
    parser.add_argument('-at', '--attributes', help = 'attributes file name',
        default = 'attributes.txt')
    parser.add_argument('-c', '--class_analysis', help = 'age class to \
        calculate mean and standard deviation for each CpG-site (by default \
        for all ages)', type = int, default = 0)
    parser.add_argument('-q', '--quiet', help = 'silent mode (save histograms)',
        action = 'store_true')
    parser.add_argument('-hcs', '--hist_classes',
        help = 'file name to save histogram of patient distribution by age',
        default = './diagrams/hist_class_samples.pdf')
    parser.add_argument('-hm', '--hist_mean_std',
        help = 'file name to save histogram of mean/std values for the specific age class',
        default = './diagrams/hist_mean_std_{}.pdf')
    args = parser.parse_args()

    [metilation_values, cpg_cites] = load_data(args.metilation)
    person_ages = load_attributes(args.attributes)
    
    hist_class_samples(person_ages, args.quiet, args.hist_classes)
    hist_mean_std(metilation_values, person_ages, args.class_analysis,
         args.quiet, args.hist_mean_std)
    