import numpy as np
import math

IMBALANCE_DIST=['linear', 'shuffled_linear', 'step', 'shuffled_step', 'random', 'random_step', 'random_capped', 'random_sorted', 'constant_controlled', 'balanced', 'tail', 'unbiased', 'shuffled_unbiased']

__cache = None
    
    
def calc_comb(n,k):
    return int(math.factorial(n+k-1) / (math.factorial(n-1) * math.factorial(k)))

def combination_generator(i,n,k):
    """
    returns the i-th combination of k numbers chosen from 1,2,...,n
    """
#     all_combs = calc_comb(n, k)
#     assert 0<= i < all_combs, "ith index must be within range 0 <= i < {}".format(calc_comb(n, k))
    
    combs = []
    remaining = i
    digit_proposal = 1

    for place in range(k):
        digit_combs = calc_comb(n-digit_proposal+1, k-1-place)
        while remaining - digit_combs >= 0:
            remaining -= digit_combs
            digit_proposal += 1
            digit_combs = calc_comb(n-digit_proposal+1, k-1-place)
        combs.append(digit_proposal)
    return np.array(combs)

def get_num_samples_per_class(imbalance_distribution, num_classes, min_num_samples, max_num_samples, num_minority, rng):
    
    # Linear distribution of k-shots
    if imbalance_distribution in ['linear', None, 'shuffled_linear']:
        num_samples = np.linspace(min_num_samples-0.49, max_num_samples+0.49, num_classes)
        num_samples = np.around(num_samples).astype(int) # round to nearest int

    # Step imbalance, controlled by num_minority
    elif imbalance_distribution in ['step', 'shuffled_step']:
        num_minority = int(num_minority * num_classes)
        num_samples = np.empty(num_classes, dtype=int)
        num_samples[:num_minority] = min_num_samples
        num_samples[num_minority:] = max_num_samples
        
    # Random distribution of k-shots between max and min
    elif imbalance_distribution in ['random', 'random_sorted']:
        num_samples = rng.randint(min_num_samples, max_num_samples+1, num_classes)
    
    # Uniform Random Imbalance distribution of k-shots between max and min, this distribution is automatically sorted
    elif imbalance_distribution in ['unbiased', 'shuffled_unbiased']:
        n = max_num_samples - min_num_samples + 1
        k = num_classes
        ith = rng.randint(calc_comb(n,k))
        num_samples = combination_generator(ith,n,k) + min_num_samples - 1

    # Step imbalance, controlled by a randomly selected num_minority
    elif imbalance_distribution == 'random_step':
        num_minority = rng.randint(1, num_classes)  # minimum 1 minority class & 1 majority class
        num_samples = np.empty(num_classes, dtype=int)
        num_samples[:num_minority] = min_num_samples
        num_samples[num_minority:] = max_num_samples

    # Random imbalance with capped total samples
    elif imbalance_distribution == 'random_capped':
        num_samples = np.ones(num_classes, dtype=int) # at least one sample per class
        total = int((min_num_samples + max_num_samples)/2) * num_classes # capped total
        unique, counts = np.unique(rng.choice(num_classes, total - num_classes), return_counts=True)
        num_samples[unique] += counts

    # Constant (specific) imbalance, with capped total samples
    elif imbalance_distribution == 'constant_controlled':
        total = int((min_num_samples + max_num_samples)/2) * num_classes  # capped total
        found = False
        while not found:
            num_samples = np.ones(num_classes, dtype=int)  # at least one sample per class
            num_samples[0] = min_num_samples
            num_samples[num_classes-1] = max_num_samples
            unique, counts = np.unique(rng.choice(
                num_classes - 2,
                total - min_num_samples - max_num_samples - num_classes + 2, 
                return_counts=True))
            num_samples[unique+1] += counts
            if np.all(num_samples) <= max_num_samples:
                found=True
    
    # No imbalance
    elif imbalance_distribution == 'balanced':
        mean_samples = int((min_num_samples+max_num_samples)/2)
        num_samples = np.ones(num_classes, dtype=int) * mean_samples
    
    # Long-tail distribution, where min_minority is used as the exponential for power law distribution
    elif imbalance_distribution == 'tail':
        px,py = (1./num_classes, min_num_samples/max_num_samples)
        qx,qy = (1., 1.)
        
        # power law distribution: y = ax^b + c
        # where y is the fraction of samples per class, and x is the class id.
        b = num_minority
        c = py
        a = (qy - c)/(qx**b)
        
        x = np.linspace(0,1,num_classes)
        y = (a*(x**b) + c)
        num_samples = (y * max_num_samples).astype(np.int32)
        
    else:
        raise Exception("Imbalance distribution not found: {}".format(imbalance_distribution))
    
    if imbalance_distribution in ['shuffled_linear', 'shuffled_step', 'shuffled_unbiased']:
        np.random.shuffle(num_samples)
    
    if imbalance_distribution in ['random_sorted']:
        num_samples.sort()
    
    return num_samples