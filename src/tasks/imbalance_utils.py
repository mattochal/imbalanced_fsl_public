import numpy as np


IMBALANCE_DIST=['linear', '__linear', 'step', '__step', 'random', 'random_step', 'random_capped', 
                'constant_controlled', 'balanced']
    

def get_num_samples_per_class(imbalance_distribution, num_classes, min_num_samples, max_num_samples, num_minority, rng):
    
    # Linear distribution of k-shots
    if imbalance_distribution in ['linear', None, '__linear']:
        num_samples = np.linspace(min_num_samples-0.49, max_num_samples+0.49, num_classes)
        num_samples = np.around(num_samples).astype(int) # round to nearest int

    # Step imbalance, controlled by num_minority
    elif imbalance_distribution in ['step', '__step']:
        num_minority = int(num_minority * num_classes)
        num_samples = np.empty(num_classes, dtype=int)
        num_samples[:num_minority] = min_num_samples
        num_samples[num_minority:] = max_num_samples
        
    # Random distribution of k-shots, between max and min
    elif imbalance_distribution == 'random':
        num_samples = rng.randint(min_num_samples, max_num_samples+1, num_classes)

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
        
    else:
        raise Exception("Imbalance distribution not found: {}".format(imbalance_distribution))
    
    if imbalance_distribution in ['__linear', '__step']:
        np.random.shuffle(num_samples)
    
    return num_samples