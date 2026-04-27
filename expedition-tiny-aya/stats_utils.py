import numpy as np

def bootstrap_ci(scores, n_resamples=10000, ci=0.95):
    scores=np.array(scores)
    result=[]

    for i in range(n_resamples):
        sample = np.random.choice(scores, size=len(scores), replace=True)

        result.append(sample.mean())

    lower = np.percentile(result, 2.5)  
    upper = np.percentile(result, 97.5)
    
    return (scores.mean(), lower, upper)


def paired_bootstrap_test(score_A,score_B, n_resamples=10000):
    scores_a = np.array(score_A)
    scores_b = np.array(score_B)
    actual_diff=scores_a.mean() - scores_b.mean()
    combined_array=np.concatenate([score_A,score_B])

    counter=0

    for i in range(n_resamples):
        sample = np.random.choice(combined_array, size=len(combined_array), replace=True)
        
        fake_a=sample[:len(score_A)]
        fake_b=sample[len(score_A):]

        fake_diff=fake_a.mean()-fake_b.mean()

        if (fake_diff >= actual_diff):
            counter+=1

    
    p_value=counter/n_resamples

    return p_value

def holm_bonferroni(p_values, alpha=0.05):
    total_values=len(p_values)

    sorted_values=sorted(enumerate(p_values), key=lambda x:x[1])

    result=[False]*total_values

    for rank, (original_index, p_val) in enumerate(sorted_values):
        threshold=alpha/(total_values-rank)

        if p_val<=threshold:
            result[original_index]=True
        
        else:
            break

    return result
    

def aggregate_seeds(per_seed_scores):
    per_seed_mean={}

    for value, scores in per_seed_scores.items():
        per_seed_mean[value]=np.mean(scores)
    
    all_means=list(per_seed_mean.values())

    overall_mean=np.mean(all_means)
    overall_std=np.std(all_means)

    all_scores_flat = []
    for scores in per_seed_scores.values():
        all_scores_flat.extend(scores)

    ci=bootstrap_ci(all_scores_flat)
    return {
        "mean": overall_mean,
        "std": overall_std,
        "per_seed_means": per_seed_mean,
        "bootstrap_ci": ci
    }       

def cohen_d(scores_a, scores_b):
    
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    mean_a = scores_a.mean()
    mean_b = scores_b.mean()
    
    var_a = np.var(scores_a, ddof=1)   
    var_b = np.var(scores_b, ddof=1)    
    
    pooled_sd = np.sqrt((var_a + var_b) / 2)  
    
    d = (mean_b - mean_a) / pooled_sd          
    
    return d
