import pandas as pd
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
from scipy import stats
from scipy.special import logsumexp
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def get_benchmarker_name(benchmarker_id):
    """
    Map benchmarker ID to human-readable name.
    
    Parameters:
    -----------
    benchmarker_id : str or int
        The benchmarker ID
    
    Returns:
    --------
    str : The human-readable name
    """
    id_str = str(benchmarker_id)
    name_map = {
        '1': 'Expert 1',
        '2': 'Expert 2',
        '3': 'Expert 3',
        '4': 'Expert 4',
        '5': 'FlowMOP',
        '6': 'FlowCut',
        '7': 'PeacoQC'
    }
    return name_map.get(id_str, f'B{id_str}')


def aggregate_pairwise_wins(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    """
    Aggregate pairwise wins across tests (not within tests).
    This respects the dependency structure of rankings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where columns are tests, rows are ranking positions,
        values are benchmarker IDs
    
    Returns:
    --------
    win_matrix : np.ndarray
        Matrix where entry [i,j] = number of tests where i beat j
    benchmarkers : List[int]
        List of benchmarker IDs
    """
    # Get all unique benchmarkers
    all_benchmarkers = set()
    for col in df.columns:
        all_benchmarkers.update(df[col].dropna().astype(int).unique())
    benchmarkers = sorted(list(all_benchmarkers))
    n_benchmarkers = len(benchmarkers)
    
    # Create benchmarker to index mapping
    b_to_idx = {bid: idx for idx, bid in enumerate(benchmarkers)}
    
    # Initialize win matrix
    win_matrix = np.zeros((n_benchmarkers, n_benchmarkers))
    
    # Count wins across tests
    for test_col in df.columns:
        test_ranking = df[test_col].dropna()
        
        # For each pair in this test, record who won
        for i, winner_id in enumerate(test_ranking):
            winner_idx = b_to_idx[int(winner_id)]
            for j in range(i + 1, len(test_ranking)):
                loser_id = int(test_ranking.iloc[j])
                loser_idx = b_to_idx[loser_id]
                win_matrix[winner_idx, loser_idx] += 1
    
    return win_matrix, benchmarkers


def plackett_luce_log_likelihood(abilities: np.ndarray,
                                 rankings_list: List[List[int]],
                                 benchmarker_to_idx: Dict[int, int]) -> float:
    """
    Calculate log-likelihood for Plackett-Luce model.
    This properly models the probability of observing complete rankings.
    
    P(ranking) = ∏ᵢ exp(ability[ranked[i]]) / Σⱼ∈remaining exp(ability[j])
    
    Parameters:
    -----------
    abilities : np.ndarray
        Ability parameters for each benchmarker
    rankings_list : List[List[int]]
        List of rankings from each test
    benchmarker_to_idx : Dict[int, int]
        Mapping from benchmarker ID to index
    
    Returns:
    --------
    Log-likelihood value
    """
    log_lik = 0.0
    
    for ranking in rankings_list:
        # Convert IDs to indices
        ranking_indices = [benchmarker_to_idx[bid] for bid in ranking]
        
        # Calculate log probability for this ranking
        for pos, chosen_idx in enumerate(ranking_indices[:-1]):  # Don't need last position
            # Get abilities of remaining items (including chosen)
            remaining_indices = ranking_indices[pos:]
            remaining_abilities = abilities[remaining_indices]
            
            # Log probability of choosing this item from remaining
            log_prob = abilities[chosen_idx] - logsumexp(remaining_abilities)
            log_lik += log_prob
    
    return log_lik


def bradley_terry_aggregated_log_likelihood(abilities: np.ndarray,
                                           win_matrix: np.ndarray) -> float:
    """
    Calculate log-likelihood for Bradley-Terry model with aggregated wins.
    This respects the test-level structure of the data.
    
    Parameters:
    -----------
    abilities : np.ndarray
        Ability parameters for each benchmarker
    win_matrix : np.ndarray
        Matrix of win counts between benchmarkers
    
    Returns:
    --------
    Log-likelihood value
    """
    n_benchmarkers = len(abilities)
    log_lik = 0.0
    
    for i in range(n_benchmarkers):
        for j in range(i + 1, n_benchmarkers):
            n_ij = win_matrix[i, j]  # Times i beat j
            n_ji = win_matrix[j, i]  # Times j beat i
            
            if n_ij + n_ji > 0:  # If they ever competed
                # Log probability of observing n_ij wins for i out of (n_ij + n_ji) trials
                ability_diff = abilities[i] - abilities[j]
                log_p_i_wins = ability_diff - np.log(1 + np.exp(ability_diff))
                log_p_j_wins = -ability_diff - np.log(1 + np.exp(-ability_diff))
                
                log_lik += n_ij * log_p_i_wins + n_ji * log_p_j_wins
    
    return log_lik


class LogPosteriorFunction:
    """
    Callable class for computing log posterior probability.
    This class encapsulates all data needed for MCMC sampling and can be pickled
    for multiprocessing.
    """
    
    def __init__(self, model_type: str, n_benchmarkers: int, ref_idx: int,
                 prior_std: float, rankings_list: Optional[List[List[int]]] = None,
                 benchmarker_to_idx: Optional[Dict[int, int]] = None,
                 win_matrix: Optional[np.ndarray] = None):
        """
        Initialize the log posterior function.
        
        Parameters:
        -----------
        model_type : str
            'plackett-luce' or 'bradley-terry-aggregated'
        n_benchmarkers : int
            Total number of benchmarkers
        ref_idx : int
            Index of reference benchmarker (ability = 0)
        prior_std : float
            Standard deviation for ability priors
        rankings_list : Optional[List[List[int]]]
            List of rankings for Plackett-Luce model
        benchmarker_to_idx : Optional[Dict[int, int]]
            Mapping from benchmarker ID to index
        win_matrix : Optional[np.ndarray]
            Win matrix for Bradley-Terry model
        """
        self.model_type = model_type
        self.n_benchmarkers = n_benchmarkers
        self.ref_idx = ref_idx
        self.prior_std = prior_std
        self.rankings_list = rankings_list
        self.benchmarker_to_idx = benchmarker_to_idx
        self.win_matrix = win_matrix
    
    def params_to_abilities(self, theta: np.ndarray) -> np.ndarray:
        """Map from parameter vector to full abilities array."""
        abilities = np.zeros(self.n_benchmarkers)
        param_idx = 0
        for i in range(self.n_benchmarkers):
            if i != self.ref_idx:
                abilities[i] = theta[param_idx]
                param_idx += 1
        # Reference stays at 0
        return abilities
    
    def log_likelihood(self, abilities: np.ndarray) -> float:
        """Compute log likelihood based on model type."""
        if self.model_type == 'plackett-luce':
            return plackett_luce_log_likelihood(abilities, self.rankings_list, 
                                               self.benchmarker_to_idx)
        elif self.model_type == 'bradley-terry-aggregated':
            return bradley_terry_aggregated_log_likelihood(abilities, self.win_matrix)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def __call__(self, theta: np.ndarray) -> float:
        """
        Compute log posterior probability.
        
        Parameters:
        -----------
        theta : np.ndarray
            Parameter vector (n_benchmarkers - 1 elements)
        
        Returns:
        --------
        Log posterior probability
        """
        # Prior: Normal(0, prior_std) for each non-reference ability
        log_prior = np.sum(stats.norm.logpdf(theta, 0, self.prior_std))
        
        # Likelihood
        abilities = self.params_to_abilities(theta)
        log_lik = self.log_likelihood(abilities)
        
        # Check for numerical issues
        if not np.isfinite(log_prior) or not np.isfinite(log_lik):
            return -np.inf
        
        return log_prior + log_lik


def run_bayesian_ranking_analysis(df: pd.DataFrame,
                                 model_type: str = 'plackett-luce',
                                 n_walkers: int = 32,
                                 n_steps: int = 5000,
                                 n_burn: int = 1000,
                                 prior_std: float = 2.0,
                                 reference_benchmarker: Optional[int] = None,
                                 n_threads: Optional[int] = None,
                                 verbose: bool = True,
                                 show_progress: bool = True,
                                 progress_chunks: int = 100) -> Dict:
    """
    Run proper Bayesian ranking analysis using MCMC.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Rankings data
    model_type : str
        'plackett-luce' (recommended) or 'bradley-terry-aggregated'
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Number of burn-in steps
    prior_std : float
        Standard deviation for ability priors
    reference_benchmarker : Optional[int]
        Which benchmarker to use as reference (ability = 0). If None, uses first.
    n_threads : Optional[int]
        Number of parallel threads for MCMC sampling. If None, defaults to 
        cpu_count() - 1 (all available CPUs minus one).
        Uses multiprocessing.Pool for parallelization.
    verbose : bool
        Print progress messages
    show_progress : bool
        Show progress bar
    progress_chunks : int
        Update progress every N steps
    
    Returns:
    --------
    Dict with MCMC results
    """
    # Get all benchmarkers
    all_benchmarkers = set()
    for col in df.columns:
        all_benchmarkers.update(df[col].dropna().astype(int).unique())
    benchmarkers = sorted(list(all_benchmarkers))
    n_benchmarkers = len(benchmarkers)
    
    if reference_benchmarker is None:
        reference_benchmarker = benchmarkers[0]
    
    if reference_benchmarker not in benchmarkers:
        raise ValueError(f"Reference benchmarker {reference_benchmarker} not in data")
    
    ref_idx = benchmarkers.index(reference_benchmarker)
    
    if verbose:
        print(f"Model type: {model_type}")
        print(f"Found {n_benchmarkers} benchmarkers: {benchmarkers}")
        print(f"Using benchmarker {reference_benchmarker} as reference (ability = 0)")
        print(f"Running MCMC: {n_walkers} walkers, {n_steps} steps, {n_burn} burn-in")
    
    # Prepare data based on model type
    benchmarker_to_idx = {bid: idx for idx, bid in enumerate(benchmarkers)}
    
    # Prepare data and create log posterior function
    if model_type == 'plackett-luce':
        # Extract rankings from each test
        rankings_list = []
        for col in df.columns:
            ranking = df[col].dropna().astype(int).tolist()
            if ranking:
                rankings_list.append(ranking)
        
        if verbose:
            print(f"Found {len(rankings_list)} test rankings")
            print(f"Average ranking length: {np.mean([len(r) for r in rankings_list]):.1f}")
        
        # Create log posterior function
        log_posterior_fn = LogPosteriorFunction(
            model_type=model_type,
            n_benchmarkers=n_benchmarkers,
            ref_idx=ref_idx,
            prior_std=prior_std,
            rankings_list=rankings_list,
            benchmarker_to_idx=benchmarker_to_idx
        )
    
    elif model_type == 'bradley-terry-aggregated':
        # Aggregate wins across tests
        win_matrix, _ = aggregate_pairwise_wins(df)
        
        if verbose:
            total_comparisons = np.sum(win_matrix)
            print(f"Aggregated win matrix: {total_comparisons:.0f} total pairwise results")
            print(f"Average wins per benchmarker: {np.sum(win_matrix, axis=1).mean():.1f}")
        
        # Create log posterior function
        log_posterior_fn = LogPosteriorFunction(
            model_type=model_type,
            n_benchmarkers=n_benchmarkers,
            ref_idx=ref_idx,
            prior_std=prior_std,
            win_matrix=win_matrix
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set up the parameter space (n-1 free parameters, reference is fixed at 0)
    n_params = n_benchmarkers - 1
    
    # Initialize walkers
    pos = np.random.randn(n_walkers, n_params) * 0.5  # Start reasonably close to 0
    
    # Set default number of threads if not specified
    if n_threads is None:
        n_threads = max(1, cpu_count() - 1)
    
    # Set up sampler with optional parallelization
    if n_threads > 1:
        if verbose:
            print(f"Using {n_threads} threads for parallel MCMC sampling")
        with Pool(processes=n_threads) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior_fn, pool=pool)
            
            if verbose:
                print("\nRunning MCMC with parallelization...")
            
            # Run MCMC with pool
            if show_progress and progress_chunks < n_steps:
                from tqdm import tqdm
                
                n_chunks = n_steps // progress_chunks
                remainder = n_steps % progress_chunks
                
                pbar = tqdm(total=n_steps, desc="MCMC Sampling", unit="steps")
                
                state = pos
                for i in range(n_chunks):
                    state = sampler.run_mcmc(state, progress_chunks, progress=False)
                    pbar.update(progress_chunks)
                    
                    if (i + 1) % 10 == 0:
                        mean_acc = np.mean(sampler.acceptance_fraction)
                        pbar.set_postfix({'Accept': f'{mean_acc:.3f}'})
                
                if remainder > 0:
                    state = sampler.run_mcmc(state, remainder, progress=False)
                    pbar.update(remainder)
                
                pbar.close()
            else:
                sampler.run_mcmc(pos, n_steps, progress=False)
    else:
        # Run with single thread
        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior_fn)
        
        if verbose:
            print("\nRunning MCMC (single-threaded)...")
        
        # Run MCMC
        if show_progress and progress_chunks < n_steps:
            from tqdm import tqdm
            
            n_chunks = n_steps // progress_chunks
            remainder = n_steps % progress_chunks
            
            pbar = tqdm(total=n_steps, desc="MCMC Sampling", unit="steps")
            
            state = pos
            for i in range(n_chunks):
                state = sampler.run_mcmc(state, progress_chunks, progress=False)
                pbar.update(progress_chunks)
                
                if (i + 1) % 10 == 0:
                    mean_acc = np.mean(sampler.acceptance_fraction)
                    pbar.set_postfix({'Accept': f'{mean_acc:.3f}'})
            
            if remainder > 0:
                state = sampler.run_mcmc(state, remainder, progress=False)
                pbar.update(remainder)
            
            pbar.close()
        else:
            sampler.run_mcmc(pos, n_steps, progress=False)
    
    if verbose:
        mean_acc = np.mean(sampler.acceptance_fraction)
        print(f"\nMCMC complete!")
        print(f"Mean acceptance rate: {mean_acc:.3f}")
        
        if mean_acc < 0.2:
            print("⚠️  Low acceptance rate - consider increasing prior_std")
        elif mean_acc > 0.8:
            print("⚠️  High acceptance rate - consider decreasing prior_std")
        else:
            print("✓ Good acceptance rate")
    
    # Extract samples
    samples_raw = sampler.get_chain(discard=n_burn, flat=True)
    
    # Convert to full ability samples
    n_samples = samples_raw.shape[0]
    samples = np.zeros((n_samples, n_benchmarkers))
    for s in range(n_samples):
        samples[s] = log_posterior_fn.params_to_abilities(samples_raw[s])
    
    # Calculate effective sample size
    try:
        from emcee import autocorr
        tau = autocorr.integrated_time(samples_raw, quiet=True)
        n_eff = n_samples / np.mean(tau)
        if verbose:
            print(f"Effective sample size: ~{n_eff:.0f} ({n_eff/n_samples*100:.1f}% efficiency)")
    except:
        n_eff = None
    
    return {
        'samples': samples,
        'samples_raw': samples_raw,
        'benchmarkers': benchmarkers,
        'benchmarker_to_idx': benchmarker_to_idx,
        'reference_benchmarker': reference_benchmarker,
        'reference_idx': ref_idx,
        'sampler': sampler,
        'model_type': model_type,
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'n_burn': n_burn,
        'n_eff': n_eff,
        'acceptance_fraction': sampler.acceptance_fraction
    }


def analyze_posterior(results: Dict,
                      target_benchmarker: int,
                      compare_against: List[int],
                      test_type: str,
                      credible_level: float = 0.95,
                      verbose: bool = True,
                      validation: Optional[Dict] = None) -> Dict:
    """
    Analyze posterior distributions for hypothesis testing.
    Computes directional Bayes factors (H1/H0) from posterior odds
    with equal priors, alongside summary stats of ability differences.
    
    Parameters:
    -----------
    results : Dict
        Results from run_bayesian_ranking_analysis
    target_benchmarker : int
        Target benchmarker to test
    compare_against : List[int]
        Benchmarkers to compare against
    test_type : str
        'superiority' or 'inferiority'
    credible_level : float
        Credible interval level
    verbose : bool
        Print results
    
    Returns:
    --------
    Dict with analysis results, including per-comparison:
    - 'bayes_factor': BF10 for the specified direction
    - 'probability': posterior probability for the direction (for reference)
    - 'evidence_strength': textual interpretation of BF (Jeffreys' scale)
    """
    samples = results['samples']
    benchmarkers = results['benchmarkers']
    benchmarker_to_idx = results['benchmarker_to_idx']
    
    if test_type not in ['superiority', 'inferiority']:
        raise ValueError("test_type must be 'superiority' or 'inferiority'")
    
    target_idx = benchmarker_to_idx[target_benchmarker]
    
    # Calculate posterior statistics
    posterior_stats = []
    for i, bid in enumerate(benchmarkers):
        abilities = samples[:, i]
        posterior_stats.append({
            'benchmarker': bid,
            'mean': np.mean(abilities),
            'median': np.median(abilities),
            'std': np.std(abilities),
            f'{credible_level:.0%}_CI_lower': np.percentile(abilities, (1-credible_level)/2*100),
            f'{credible_level:.0%}_CI_upper': np.percentile(abilities, (1+credible_level)/2*100)
        })
    
    posterior_df = pd.DataFrame(posterior_stats)
    posterior_df = posterior_df.sort_values('mean', ascending=False)
    
    if verbose:
        print(f"\nPosterior Statistics (Reference: {get_benchmarker_name(results['reference_benchmarker'])} = 0)")
        print("="*70)
        print(posterior_df.to_string(index=False))

        # Convergence diagnostics from MCMC sampler
        print(f"\nMCMC DIAGNOSTICS")
        print("="*70)
        acc_frac = results.get('acceptance_fraction', None)
        if acc_frac is not None and len(acc_frac) > 0:
            mean_acc = np.mean(acc_frac)
            min_acc = np.min(acc_frac)
            max_acc = np.max(acc_frac)
            print(f"Mean acceptance fraction: {mean_acc:.3f}")
            print(f"Min / Max acceptance fraction: {min_acc:.3f} / {max_acc:.3f}")
        else:
            print("Mean acceptance fraction: N/A")

        n_eff = results.get('n_eff', None)
        if n_eff is not None:
            print(f"Approximate effective sample size: {n_eff:.0f}")
        else:
            print("Approximate effective sample size: N/A")

        # Posterior predictive model fit diagnostics, if provided
        if validation is not None:
            chi2_stat = validation.get('chi2_stat', None)
            n_comp = validation.get('n_comparisons', None)
            avg_chi2 = validation.get('avg_chi2_per_comparison', None)
            fit_class = validation.get('fit_classification', None)

            if chi2_stat is not None and n_comp is not None and avg_chi2 is not None:
                print(f"\nPOSTERIOR PREDICTIVE MODEL FIT (PAIRWISE WINS)")
                print("="*70)
                print(f"Chi-square statistic: {chi2_stat:.2f}")
                print(f"Number of pairwise comparisons: {n_comp}")
                print(f"Average chi-square per comparison: {avg_chi2:.2f}")
                if fit_class is not None:
                    print(f"Fit assessment: {fit_class}")
    
    # Test specific hypotheses
    test_results = {}
    
    for comp_id in compare_against:
        if comp_id not in benchmarker_to_idx:
            continue
            
        comp_idx = benchmarker_to_idx[comp_id]
        
        # Calculate posterior probability for the directional hypothesis
        if test_type == 'superiority':
            prob = np.mean(samples[:, target_idx] > samples[:, comp_idx])
        else:  # inferiority
            prob = np.mean(samples[:, target_idx] < samples[:, comp_idx])

        # Convert posterior probability to Bayes factor (posterior odds with equal priors)
        # BF10 = P(H1 | data) / P(H0 | data) with prior odds = 1
        eps = 1e-9
        p_clipped = np.clip(prob, eps, 1 - eps)
        bayes_factor = p_clipped / (1 - p_clipped)
        
        # Calculate difference
        diff = samples[:, target_idx] - samples[:, comp_idx]
        
        test_results[comp_id] = {
            'probability': prob,  # retained for reference
            'bayes_factor': bayes_factor,
            'diff_mean': np.mean(diff),
            'diff_median': np.median(diff),
            'diff_std': np.std(diff),
            f'diff_{credible_level:.0%}_CI': [
                np.percentile(diff, (1-credible_level)/2*100),
                np.percentile(diff, (1+credible_level)/2*100)
            ],
            'evidence_strength': interpret_bayes_factor(bayes_factor)
        }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"{test_type.upper()} TEST RESULTS")
        print(f"Target: {get_benchmarker_name(target_benchmarker)}")
        print(f"Testing against: {[get_benchmarker_name(c) for c in compare_against]}")
        print("="*70)
        
        for comp_id, result in test_results.items():
            print(f"\n{get_benchmarker_name(target_benchmarker)} vs {get_benchmarker_name(comp_id)}:")
            symbol = '>' if test_type == 'superiority' else '<'
            print(f"  BF({get_benchmarker_name(target_benchmarker)} {symbol} {get_benchmarker_name(comp_id)}): {result['bayes_factor']:.2f}")
            print(f"  Evidence: {result['evidence_strength']}  (P={result['probability']:.3f})")
            print(f"  Ability difference: {result['diff_mean']:.3f} ± {result['diff_std']:.3f}")
            ci = result[f'diff_{credible_level:.0%}_CI']
            print(f"  {credible_level:.0%} CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    return {
        'posterior_stats': posterior_df,
        'test_results': test_results,
        'test_type': test_type,
        'target': target_benchmarker,
        'compare_against': compare_against,
        'validation': validation
    }


def interpret_bayes_factor(bf: float) -> str:
    """
    Interpret Bayes factor (H1/H0) using Jeffreys' scale.
    """
    if bf >= 100:
        return "Decisive evidence"
    elif bf >= 30:
        return "Very strong evidence"
    elif bf >= 10:
        return "Strong evidence"
    elif bf >= 3:
        return "Moderate evidence"
    elif bf >= 1:
        return "Anecdotal evidence"
    elif bf > 0:
        return "Evidence for H0"
    else:
        return "No evidence"


def plot_posterior_differences(results: Dict,
                             target: int,
                             compare_against: List[int],
                             test_type: str = 'superiority',
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a professional visualization of posterior differences.
    
    Parameters:
    -----------
    results : Dict
        Results from run_bayesian_ranking_analysis
    target : int
        Target benchmarker to test
    compare_against : List[int]
        Benchmarkers to compare against
    test_type : str
        'superiority' or 'inferiority'
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    # Set seaborn style for consistency
    sns.set_style("whitegrid", {
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'grid.color': '#EEEEEE',
        'grid.linewidth': 0.5
    })
    font_scale = 1.3
    sns.set_context("notebook", font_scale=font_scale)

    def scaled(size: float) -> float:
        """Utility to consistently scale font sizes."""
        return size * font_scale
    
    samples = results['samples']
    benchmarker_to_idx = results['benchmarker_to_idx']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Calculate differences
    target_idx = benchmarker_to_idx[target]
    differences_data = []
    labels = []
    
    for i, comp_id in enumerate(compare_against):
        comp_idx = benchmarker_to_idx[comp_id]
        diff = samples[:, target_idx] - samples[:, comp_idx]
        differences_data.append(diff)
        labels.append(f'{get_benchmarker_name(target)} - {get_benchmarker_name(comp_id)}')
    
    # Create violin plot for cleaner appearance (without means and extremities)
    parts = ax.violinplot(differences_data, positions=range(len(differences_data)),
                          widths=0.7, showmeans=False, showmedians=True, 
                          showextrema=False)  # Remove whiskers/extrema
    
    # Style the violin plots based on significance
    for i, pc in enumerate(parts['bodies']):
        # Check if this comparison is significant (95% CI excludes zero)
        diff_data = differences_data[i]
        ci_lower = np.percentile(diff_data, 2.5)
        ci_upper = np.percentile(diff_data, 97.5)
        is_significant = (ci_lower > 0) or (ci_upper < 0)
        
        # Set color based on significance
        if is_significant:
            pc.set_facecolor('#C62828')  # Red for significant
        else:
            pc.set_facecolor('#2196F3')  # Blue for non-significant
        
        pc.set_alpha(0.7)
        
        # Check if this comparison involves FlowMOP (benchmarker 5)
        comp_id = compare_against[i]
        if target == 5 or comp_id == 5:
            # Bold black edge for FlowMOP comparisons
            pc.set_edgecolor('black')
            pc.set_linewidth(3)
        else:
            pc.set_edgecolor('white')
            pc.set_linewidth(1.5)
    
    # Style the median line
    parts['cmedians'].set_color('#1B5E20')  # Dark green for median
    parts['cmedians'].set_linewidth(2)
    
    # Add 95% CI as horizontal lines
    for i, diff_data in enumerate(differences_data):
        ci_lower = np.percentile(diff_data, 2.5)
        ci_upper = np.percentile(diff_data, 97.5)
        
        # Check if this comparison involves FlowMOP
        comp_id = compare_against[i]
        if target == 5 or comp_id == 5:
            # Thicker lines for FlowMOP comparisons
            ax.plot([i - 0.2, i + 0.2], [ci_lower, ci_lower], 'k-', linewidth=3, alpha=1.0)
            ax.plot([i - 0.2, i + 0.2], [ci_upper, ci_upper], 'k-', linewidth=3, alpha=1.0)
            ax.plot([i, i], [ci_lower, ci_upper], 'k-', linewidth=2.5, alpha=1.0)
        else:
            ax.plot([i - 0.2, i + 0.2], [ci_lower, ci_lower], 'k-', linewidth=2, alpha=0.8)
            ax.plot([i - 0.2, i + 0.2], [ci_upper, ci_upper], 'k-', linewidth=2, alpha=0.8)
            ax.plot([i, i], [ci_lower, ci_upper], 'k-', linewidth=1.5, alpha=0.8)
    
    # Add zero reference line
    ax.axhline(0, color='#C62828', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Customize plot
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Ability Difference', fontsize=12, fontweight='medium', color='#333333', labelpad=10)
    ax.set_xlabel('Comparison', fontsize=12, fontweight='medium', color='#333333', labelpad=10)
    
    title = f'Posterior Differences: {get_benchmarker_name(target)} {test_type.capitalize()} Test'
    ax.set_title(title, fontsize=16, fontweight='bold', color='#222222', pad=20)
    
    # Grid and spines
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.6, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Reset styles
    sns.reset_defaults()
    plt.rcdefaults()
    
    return fig


def plot_posterior_tails(results: Dict,
                        benchmarkers_to_show: Optional[List[int]] = None,
                        highlight: Optional[int] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a professional visualization of posterior 95% credible intervals.
    
    Parameters:
    -----------
    results : Dict
        Results from run_bayesian_ranking_analysis
    benchmarkers_to_show : Optional[List[int]]
        List of benchmarkers to display. If None, shows all.
    highlight : Optional[int]
        Benchmarker to highlight in different color
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'grid.color': '#EEEEEE',
        'grid.linewidth': 0.5
    })
    font_scale = 1.3
    sns.set_context("notebook", font_scale=font_scale)

    def scaled(size: float) -> float:
        """Utility to consistently scale font sizes."""
        return size * font_scale
    
    samples = results['samples']
    benchmarkers = results['benchmarkers']
    benchmarker_to_idx = results['benchmarker_to_idx']
    
    # Filter benchmarkers if specified
    if benchmarkers_to_show is None:
        benchmarkers_to_show = benchmarkers
    
    # Reorder benchmarkers: Expert methods (1-4) first, then algorithms (5, 6, 7)
    expert_methods = [b for b in benchmarkers_to_show if b in [1, 2, 3, 4]]
    algo_methods = [b for b in benchmarkers_to_show if b in [5, 6, 7]]
    # Sort each group and combine, then reverse for top-to-bottom display
    benchmarkers_to_show = sorted(expert_methods) + sorted(algo_methods)
    benchmarkers_to_show = list(reversed(benchmarkers_to_show))  # Reverse for top-to-bottom display
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Calculate statistics for each benchmarker
    y_positions = []
    labels = []
    medians = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    for i, bid in enumerate(benchmarkers_to_show):
        idx = benchmarker_to_idx[bid]
        abilities = samples[:, idx]
        
        y_positions.append(i)
        labels.append(get_benchmarker_name(bid))
        medians.append(np.median(abilities))
        ci_lowers.append(np.percentile(abilities, 2.5))
        ci_uppers.append(np.percentile(abilities, 97.5))
        
        # Color based on whether it's highlighted
        if highlight and bid == highlight:
            colors.append('#2E7D32')  # Dark green for highlighted
        else:
            colors.append('#666666')  # Gray for others
    
    # Create horizontal interval plot
    for i, (y, med, ci_l, ci_u, color) in enumerate(zip(y_positions, medians, 
                                                         ci_lowers, ci_uppers, colors)):
        bid = benchmarkers_to_show[i]
        
        # Check if this is FlowMOP (benchmarker 5)
        if bid == 5:
            # Bold black styling for FlowMOP
            # Draw CI line with black and thicker
            ax.plot([ci_l, ci_u], [y, y], color='black', linewidth=5, alpha=1.0, zorder=4)
            # Draw the colored line on top but thinner
            ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=3, alpha=0.7, zorder=5)
            
            # Draw CI endpoints with black border
            ax.plot([ci_l, ci_l], [y - 0.15, y + 0.15], color='black', linewidth=4, alpha=1.0, zorder=4)
            ax.plot([ci_u, ci_u], [y - 0.15, y + 0.15], color='black', linewidth=4, alpha=1.0, zorder=4)
            ax.plot([ci_l, ci_l], [y - 0.15, y + 0.15], color=color, linewidth=2, alpha=0.8, zorder=5)
            ax.plot([ci_u, ci_u], [y - 0.15, y + 0.15], color=color, linewidth=2, alpha=0.8, zorder=5)
            
            # Draw median point with black border
            ax.scatter(med, y, color=color, s=100, zorder=6, edgecolors='black', linewidth=3)
        else:
            # Regular styling for other benchmarkers
            # Draw CI line
            ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=3, alpha=0.7)
            
            # Draw CI endpoints
            ax.plot([ci_l, ci_l], [y - 0.15, y + 0.15], color=color, linewidth=2, alpha=0.8)
            ax.plot([ci_u, ci_u], [y - 0.15, y + 0.15], color=color, linewidth=2, alpha=0.8)
            
            # Draw median point
            ax.scatter(med, y, color=color, s=100, zorder=5, edgecolors='white', linewidth=1.5)
    
    # Add zero reference line if reference benchmarker ability = 0
    ax.axvline(0, color='#C62828', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Ability (95% Credible Interval)', fontsize=12, fontweight='medium', 
                  color='#333333', labelpad=10)
    ax.set_ylabel('Benchmarker', fontsize=12, fontweight='medium', color='#333333', labelpad=10)
    
    title = 'Posterior 95% Credible Intervals'
    ax.set_title(title, fontsize=16, fontweight='bold', color='#222222', pad=20)
    
    # Grid and spines
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.6, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add legend if there's a highlight
    if highlight:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2E7D32', linewidth=3, label=f'{get_benchmarker_name(highlight)} (Target)'),
            Line2D([0], [0], color='#666666', linewidth=3, label='Others')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    plt.tight_layout()
    
    # Reset styles
    sns.reset_defaults()
    plt.rcdefaults()
    
    return fig


def plot_posterior_analysis(results: Dict,
                           target: int,
                           compare_against: List[int],
                           test_type: str,
                           analysis_results: Optional[Dict] = None,
                           figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    [DEPRECATED] Use plot_posterior_differences and plot_posterior_tails instead.
    Visualize posterior distributions and test results.
    """
    samples = results['samples']
    benchmarkers = results['benchmarkers']
    benchmarker_to_idx = results['benchmarker_to_idx']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Relevant benchmarkers
    relevant = [target] + compare_against
    
    # Plot 1: Posterior distributions
    ax = axes[0, 0]
    for bid in relevant:
        idx = benchmarker_to_idx[bid]
        abilities = samples[:, idx]
        
        color = 'red' if bid == target else 'blue'
        alpha = 0.7 if bid == target else 0.5
        
        ax.hist(abilities, bins=50, alpha=alpha, label=get_benchmarker_name(bid), 
                density=True, color=color, edgecolor='black', linewidth=0.5)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label=f'Reference ({get_benchmarker_name(results["reference_benchmarker"])})')
    ax.set_xlabel('Ability')
    ax.set_ylabel('Posterior Density')
    ax.set_title(f'Posterior Distributions\n(Model: {results["model_type"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pairwise differences
    ax = axes[0, 1]
    target_idx = benchmarker_to_idx[target]
    
    for comp_id in compare_against:
        comp_idx = benchmarker_to_idx[comp_id]
        diff = samples[:, target_idx] - samples[:, comp_idx]
        
        ax.hist(diff, bins=50, alpha=0.5, label=f'{get_benchmarker_name(target)} - {get_benchmarker_name(comp_id)}', 
                density=True, edgecolor='black', linewidth=0.5)
    
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Ability Difference')
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior Differences ({test_type})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bayes factors
    if analysis_results:
        ax = axes[1, 0]
        test_results = analysis_results['test_results']
        
        bfs = [test_results[cid]['bayes_factor'] for cid in compare_against]
        evidence = [test_results[cid]['evidence_strength'] for cid in compare_against]
        
        def bf_color(bf: float) -> str:
            if bf >= 10:
                return 'green'
            elif bf >= 3:
                return 'orange'
            else:
                return 'red'
        colors = [bf_color(bf) for bf in bfs]
        
        bars = ax.bar(range(len(bfs)), bfs, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(bfs)))
        ax.set_xticklabels([get_benchmarker_name(cid) for cid in compare_against])
        ax.set_ylabel('Bayes Factor (H1/H0)')
        ax.set_title(f'{test_type.capitalize()} Bayes Factors')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add labels
        for bar, bf, ev in zip(bars, bfs, evidence):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.01, 0.02 * bar.get_height()),
                   f'{bf:.2f}\n{ev}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Model diagnostics
    ax = axes[1, 1]
    ax.text(0.1, 0.9, f"Model: {results['model_type']}", transform=ax.transAxes, fontweight='bold')
    ax.text(0.1, 0.8, f"Reference: {get_benchmarker_name(results['reference_benchmarker'])} (ability = 0)", transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Effective samples: {results.get('n_eff', 'N/A'):.0f}" if results.get('n_eff') else "Effective samples: N/A", transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Acceptance rate: {np.mean(results['acceptance_fraction']):.3f}", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Total samples: {len(samples)}", transform=ax.transAxes)
    
    if analysis_results:
        ax.text(0.1, 0.3, f"Test: {get_benchmarker_name(target)} {test_type}", transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.suptitle(f'Bayesian Ranking Analysis: {get_benchmarker_name(target)} {test_type} test', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_average_scores(df: pd.DataFrame,
                       figsize: Tuple[int, int] = (10, 6),
                       title: Optional[str] = None) -> plt.Figure:
    """
    Create a vertical bar chart showing average scores for each benchmarker.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Rankings data where columns are tests, rows are ranking positions,
        values are benchmarker IDs
    figsize : Tuple[int, int]
        Figure size (width, height)
    title : Optional[str]
        Custom title for the plot
    
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    # Get all unique benchmarkers
    all_benchmarkers = set()
    for col in df.columns:
        all_benchmarkers.update(df[col].dropna().astype(int).unique())
    benchmarkers = sorted(list(all_benchmarkers))
    
    # Find maximum possible ranking across all tests
    max_rank = max(len(df[col].dropna()) for col in df.columns)
    
    # Calculate average scores for each benchmarker
    average_scores = {}
    for benchmarker in benchmarkers:
        total_score = 0
        total_tests = 0
        
        for col in df.columns:
            test_ranking = df[col].dropna()
            if benchmarker in test_ranking.astype(int).values:
                position = np.where(test_ranking.astype(int).values == benchmarker)[0][0]
                rank = position + 1
                score = max_rank - rank + 1
                total_score += score
                total_tests += 1
        
        if total_tests > 0:
            average_scores[benchmarker] = total_score / total_tests
        else:
            average_scores[benchmarker] = 0
    
    # Fixed ordering: Expert 1-4, then FlowMOP, FlowCut, PeacoQC
    ordered_benchmarkers = []
    for b in [1, 2, 3, 4, 5, 6, 7]:
        if b in benchmarkers:
            ordered_benchmarkers.append(b)
    
    # Define color scheme matching chi_squared_analysis.py
    color_map = {
        1: '#808080',  # Expert 1 - gray
        2: '#808080',  # Expert 2 - gray
        3: '#808080',  # Expert 3 - gray
        4: '#808080',  # Expert 4 - gray
        5: '#4682B4',  # FlowMOP - steel blue
        6: '#D2691E',  # FlowCut - brown/sienna
        7: '#228B22'   # PeacoQC - forest green
    }
    
    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'grid.color': '#EEEEEE',
        'grid.linewidth': 0.5
    })
    font_scale = 1.3
    sns.set_context("notebook", font_scale=font_scale)

    def scaled(size: float) -> float:
        """Utility to consistently scale font sizes."""
        return size * font_scale
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Prepare data for plotting
    x_positions = np.arange(len(ordered_benchmarkers))
    scores = [average_scores[b] for b in ordered_benchmarkers]
    labels = [get_benchmarker_name(b) for b in ordered_benchmarkers]
    colors = [color_map.get(b, '#808080') for b in ordered_benchmarkers]
    
    # Create bars with individual colors
    bars = ax.bar(x_positions, scores, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Highlight FlowMOP with black border
    for i, benchmarker in enumerate(ordered_benchmarkers):
        if benchmarker == 5:  # FlowMOP
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(3)
    
    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=11, rotation=45, ha='right')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='medium', color='#333333', labelpad=10)
    ax.set_xlabel('Benchmarker', fontsize=12, fontweight='medium', color='#333333', labelpad=10)
    
    if title is None:
        title = 'Average Rankings Score'
    ax.set_title(title, fontsize=16, fontweight='bold', color='#222222', pad=20)
    
    # Grid and spines
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.6, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0, top=max(scores) * 1.1)
    
    plt.tight_layout()
    
    # Reset styles
    sns.reset_defaults()
    plt.rcdefaults()
    
    return fig


def plot_ranking_heatmap(df: pd.DataFrame,
                        figsize: Tuple[int, int] = (12, 8),
                        colors: Optional[List[str]] = None,
                        title: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing rankings across datasets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Rankings data where columns are tests, rows are ranking positions,
        values are benchmarker IDs
    figsize : Tuple[int, int]
        Figure size (width, height)
    colors : Optional[List[str]]
        Custom colors for each ranking position. If None, uses default palette
    title : Optional[str]
        Custom title for the plot
    
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    # Get all unique benchmarkers
    all_benchmarkers = set()
    for col in df.columns:
        all_benchmarkers.update(df[col].dropna().astype(int).unique())
    
    # Order benchmarkers: Expert 1-4 at top, then algorithms at bottom
    # Explicit order from top to bottom: Expert 1, 2, 3, 4, FlowMOP, FlowCut, PeacoQC
    benchmarkers = []
    for b in [1, 2, 3, 4, 5, 6, 7]:
        if b in all_benchmarkers:
            benchmarkers.append(b)
    # Reverse for display (matplotlib plots from bottom to top)
    benchmarkers = list(reversed(benchmarkers))
    
    # Get all datasets
    datasets = df.columns.tolist()
    
    # Find maximum ranking
    max_rank = max(len(df[col].dropna()) for col in df.columns)
    
    # Create matrix for heatmap (benchmarkers x datasets)
    ranking_matrix = np.full((len(benchmarkers), len(datasets)), np.nan)
    
    for j, dataset in enumerate(datasets):
        test_ranking = df[dataset].dropna()
        for position, benchmarker_id in enumerate(test_ranking):
            benchmarker_id = int(benchmarker_id)
            if benchmarker_id in benchmarkers:
                i = benchmarkers.index(benchmarker_id)
                ranking_matrix[i, j] = position + 1  # 1-indexed ranking
    
    # Set up colors using a colour-blind-friendly sequential palette
    if colors is None:
        colors = sns.color_palette("inferno", n_colors=max_rank)
    
    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'grid.color': '#EEEEEE',
        'grid.linewidth': 0.5
    })
    font_scale = 1.3
    sns.set_context("notebook", font_scale=font_scale)

    def scaled(size: float) -> float:
        """Utility to consistently scale font sizes."""
        return size * font_scale
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Create the heatmap using patches
    for i, benchmarker in enumerate(benchmarkers):
        for j, dataset in enumerate(datasets):
            rank = ranking_matrix[i, j]
            
            if not np.isnan(rank):
                rank_int = int(rank)

                # Flip the displayed rank so that 1 is worst and max_rank is best
                display_rank = max_rank - rank_int + 1

                # Use flipped rank to choose colour so that colour and displayed number align
                color_idx = display_rank - 1
                color_idx = max(0, min(color_idx, len(colors) - 1))

                # Draw colored rectangle for the rank
                rect = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                    facecolor=colors[color_idx],
                                    edgecolor='white', linewidth=2)
                ax.add_patch(rect)

                # Add flipped rank number with black text and a softer white halo
                text = ax.text(j, i, str(display_rank), ha='center', va='center',
                               color='black', fontsize=scaled(10), fontweight='bold')
                text.set_path_effects([
                    patheffects.Stroke(linewidth=3, foreground='white', alpha=0.5),
                    patheffects.Stroke(linewidth=5, foreground='white', alpha=0.25),
                    patheffects.Normal()
                ])
            else:
                # Draw empty rectangle for N/A
                rect = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                    facecolor='white',
                                    edgecolor='#CCCCCC', linewidth=1)
                ax.add_patch(rect)
                
                # Add N/A text
                ax.text(j, i, 'N/A', ha='center', va='center',
                       color='#999999', fontsize=scaled(9), style='italic')
    
    # Highlight FlowMOP row with black border
    flowmop_idx = None
    for i, b in enumerate(benchmarkers):
        if b == 5:  # FlowMOP
            flowmop_idx = i
            break
    
    if flowmop_idx is not None:
        # Draw thick black border around FlowMOP row, padded but centered on cells
        pad = 0.05
        rect = plt.Rectangle(
            (-0.4 - pad, flowmop_idx - 0.4 - pad),
            len(datasets) - 0.2 + 2 * pad,
            0.8 + 2 * pad,
            linewidth=3,
            edgecolor='black',
            facecolor='none',
            zorder=10
        )
        ax.add_patch(rect)
    
    # Set ticks and labels
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(benchmarkers)))
    ax.set_yticklabels([get_benchmarker_name(b) for b in benchmarkers], fontsize=scaled(11))
    
    # Move x-axis labels to top and make them vertical
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(
        datasets,
        fontsize=scaled(10),
        rotation=45,
        ha='left',
        va='bottom',
        rotation_mode='anchor'
    )
    
    # Set limits
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(-0.5, len(benchmarkers) - 0.5)
    
    # Labels and title
    # No x-label since column names are self-explanatory
    ax.set_ylabel('Benchmarker', fontsize=scaled(12), fontweight='medium', color='#333333', labelpad=10)
    
    if title is None:
        title = 'Rankings Across Datasets'
    # Moderate padding for title
    ax.set_title(title, fontsize=scaled(16), fontweight='bold', color='#222222', pad=20)
    
    # Remove grid for cleaner look
    ax.grid(False)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Create legend showing full gradient (unlabeled blocks)
    legend_elements = []
    if len(colors) > 0:
        for c in colors:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=c, label=''))
    
    legend = ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(legend_elements) if legend_elements else 1,
        frameon=False,
        fontsize=scaled(10),
        handlelength=1.0,
        handletextpad=0.0,
        borderaxespad=0.2
    )

    # Add "Worst" and "Best" labels beneath the gradient blocks,
    # horizontally aligned with the first and last legend patches.
    if len(colors) > 0:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
        bbox_ax = bbox.transformed(ax.transAxes.inverted())

        x_left = bbox_ax.x0
        x_right = bbox_ax.x1
        y_bottom = bbox_ax.y0
        pad = 0.05

        ax.text(
            x_left, y_bottom - pad, "Worst",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=scaled(10)
        )
        ax.text(
            x_right, y_bottom - pad, "Best",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=scaled(10)
        )
    
    # Adjust layout with reduced top spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Reset styles
    sns.reset_defaults()
    plt.rcdefaults()
    
    return fig


def validate_model_fit(results: Dict, df: pd.DataFrame, 
                      n_posterior_samples: int = 100,
                      verbose: bool = True) -> Dict:
    """
    Validate model fit using posterior predictive checks.
    
    Parameters:
    -----------
    results : Dict
        Results from run_bayesian_ranking_analysis
    df : pd.DataFrame
        Original data
    n_posterior_samples : int
        Number of posterior samples to use
    verbose : bool
        Print validation results
    
    Returns:
    --------
    Dict with validation metrics
    """
    samples = results['samples']
    benchmarkers = results['benchmarkers']
    benchmarker_to_idx = results['benchmarker_to_idx']
    
    # Randomly select posterior samples
    sample_indices = np.random.choice(len(samples), n_posterior_samples, replace=False)
    
    # Calculate observed win rates
    win_matrix_obs, _ = aggregate_pairwise_wins(df)
    
    # Generate predicted win probabilities
    n_benchmarkers = len(benchmarkers)
    win_probs_pred = np.zeros((n_benchmarkers, n_benchmarkers))
    
    for s_idx in sample_indices:
        abilities = samples[s_idx]
        for i in range(n_benchmarkers):
            for j in range(n_benchmarkers):
                if i != j:
                    # P(i beats j) under Bradley-Terry
                    prob = 1 / (1 + np.exp(abilities[j] - abilities[i]))
                    win_probs_pred[i, j] += prob
    
    win_probs_pred /= n_posterior_samples
    
    # Compare observed vs predicted
    n_tests = len(df.columns)
    
    # Calculate chi-square-like statistic
    chi2_stat = 0
    n_comparisons = 0
    
    for i in range(n_benchmarkers):
        for j in range(i+1, n_benchmarkers):
            n_ij = win_matrix_obs[i, j]
            n_ji = win_matrix_obs[j, i]
            n_total = n_ij + n_ji
            
            if n_total > 0:
                # Expected wins
                exp_ij = n_total * win_probs_pred[i, j]
                exp_ji = n_total * win_probs_pred[j, i]
                
                # Chi-square contribution (with continuity correction)
                if exp_ij > 0:
                    chi2_stat += (n_ij - exp_ij)**2 / exp_ij
                if exp_ji > 0:
                    chi2_stat += (n_ji - exp_ji)**2 / exp_ji
                
                n_comparisons += 1

    # Guard against division by zero
    avg_chi2_per_comparison = chi2_stat / n_comparisons if n_comparisons > 0 else np.nan

    # Classify model fit using the same thresholds used in printing
    if np.isnan(avg_chi2_per_comparison):
        fit_classification = "Insufficient comparisons to assess fit"
    elif avg_chi2_per_comparison < 2:
        fit_classification = "Good model fit"
    elif avg_chi2_per_comparison < 4:
        fit_classification = "Moderate model fit"
    else:
        fit_classification = "Poor model fit - model may be misspecified"

    if verbose:
        print("\nMODEL VALIDATION")
        print("="*50)
        print(f"Chi-square statistic: {chi2_stat:.2f}")
        print(f"Number of pairwise comparisons: {n_comparisons}")
        print(f"Average chi-square per comparison: {avg_chi2_per_comparison:.2f}")

        if "Good" in fit_classification:
            symbol = "✓"
        elif "Moderate" in fit_classification:
            symbol = "⚠️"
        elif "Poor" in fit_classification:
            symbol = "✗"
        else:
            symbol = "-"
        print(f"{symbol} {fit_classification}")
    
    return {
        'chi2_stat': chi2_stat,
        'n_comparisons': n_comparisons,
        'avg_chi2_per_comparison': avg_chi2_per_comparison,
        'fit_classification': fit_classification,
        'win_matrix_observed': win_matrix_obs,
        'win_probs_predicted': win_probs_pred
    }


# Example usage
if __name__ == "__main__":
    # Create example data
    data = {
        'Mouse DRG': [3, 2, 3, 4, 5, 6, 7],
        'Mouse Skin': [2, 2, 1, 4, 5, 6, 7],
        'Human T Diff': [3, 3, 5, 4, 1, 6, 7],
        'Mouse Bonemarrow': [3, 2, 3, 1, 5, 6, 7],
        'Mouse Spleen': [3, 2, 1, 4, 5, 6, 7],
        'Mouse Blood': [5, 3, 1, 4, 5, 6, 7],
        'Mouse Brain': [3, 3, 2, 4, 1, 6, 7],
        'Mouse CNS': [2, 2, 4, 1, 5, 6, 7],
        'Human Liver': [2.0, 1.0, 5.0, 3.0, np.nan, 6.0, 7.0]
    }
    
    df = pd.DataFrame(data)
    print("Input DataFrame (Rankings):")
    print(df)
    print("\n" + "="*70)
    
    # Show aggregated win matrix
    print("\nAGGREGATED WIN MATRIX:")
    print("="*70)
    win_matrix, benchmarkers = aggregate_pairwise_wins(df)
    win_df = pd.DataFrame(win_matrix, 
                         index=[get_benchmarker_name(b) for b in benchmarkers],
                         columns=[get_benchmarker_name(b) for b in benchmarkers])
    print("Rows beat columns (count across all tests):")
    print(win_df)
    print("\nNote: This respects the test-level structure of the data.")
    print("Each cell shows how many tests the row benchmarker beat the column benchmarker.")
    
    print("\n" + "="*70)
    print("RUNNING BAYESIAN ANALYSIS WITH PLACKETT-LUCE MODEL")
    print("="*70)
    
    # Run with Plackett-Luce model (RECOMMENDED)
    results_pl = run_bayesian_ranking_analysis(
        df=df,
        model_type='plackett-luce',  # Properly models complete rankings
        n_walkers=32,
        n_steps=5000,
        n_burn=1000,
        prior_std=2.0,
        reference_benchmarker=7,  # Use worst performer as reference
        verbose=True,
        show_progress=True,
        progress_chunks=100
    )
    
    # Validate model fit
    print("\n" + "="*70)
    validation = validate_model_fit(results_pl, df, verbose=True)
    
    # Test 1: Superiority
    print("\n" + "="*70)
    print("TEST 1: Is Benchmarker 5 SUPERIOR to Benchmarkers 6 and 7?")
    print("="*70)
    
    superiority_analysis = analyze_posterior(
        results=results_pl,
        target_benchmarker=5,
        compare_against=[6, 7],
        test_type='superiority',
        credible_level=0.95,
        verbose=True
    )
    
    # Test 2: Inferiority
    print("\n" + "="*70)
    print("TEST 2: Is Benchmarker 5 INFERIOR to Benchmarkers 1, 2, 3, and 4?")
    print("="*70)
    
    inferiority_analysis = analyze_posterior(
        results=results_pl,
        target_benchmarker=5,
        compare_against=[1, 2, 3, 4],
        test_type='inferiority',
        credible_level=0.95,
        verbose=True
    )
    
    # New Professional Visualizations
    print("\n" + "="*70)
    print("POSTERIOR VISUALIZATIONS")
    print("="*70)
    
    # Visualization 1: Posterior differences for superiority test
    fig1 = plot_posterior_differences(
        results_pl, 
        target=5, 
        compare_against=[6, 7],
        test_type='superiority'
    )
    plt.show()
    
    # Visualization 2: Posterior differences for inferiority test
    fig2 = plot_posterior_differences(
        results_pl,
        target=5,
        compare_against=[1, 2, 3, 4], 
        test_type='inferiority'
    )
    plt.show()
    
    # Visualization 3: Posterior 95% credible intervals
    fig3 = plot_posterior_tails(
        results_pl,
        benchmarkers_to_show=None,  # Show all
        highlight=5  # Highlight benchmarker 5
    )
    plt.show()
    
    print("\n" + "="*70)
    print("COMPARISON WITH AGGREGATED BRADLEY-TERRY")
    print("="*70)
    
    # Run with aggregated Bradley-Terry for comparison
    results_bt = run_bayesian_ranking_analysis(
        df=df,
        model_type='bradley-terry-aggregated',
        n_walkers=32,
        n_steps=5000,
        n_burn=1000,
        prior_std=2.0,
        reference_benchmarker=7,
        verbose=True,
        show_progress=True,
        progress_chunks=100
    )
    
    # Compare results
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Extract abilities for benchmarker 5
    b5_idx = benchmarkers.index(5)
    
    pl_mean = np.mean(results_pl['samples'][:, b5_idx])
    pl_std = np.std(results_pl['samples'][:, b5_idx])
    
    bt_mean = np.mean(results_bt['samples'][:, b5_idx])
    bt_std = np.std(results_bt['samples'][:, b5_idx])
    
    print(f"Benchmarker 5 ability estimates:")
    print(f"  Plackett-Luce:          {pl_mean:.3f} ± {pl_std:.3f}")
    print(f"  Bradley-Terry (aggregated): {bt_mean:.3f} ± {bt_std:.3f}")
    print(f"\nNote: Plackett-Luce is more statistically appropriate for ranking data.")
    
    print("\n" + "="*70)
    print("RANKING PERFORMANCE VISUALIZATIONS")
    print("="*70)
    
    # Create average scores visualization
    fig_avg_scores = plot_average_scores(
        df=df,
        figsize=(10, 6),
        title="Average Rankings Score"
    )
    plt.show()
    
    # Create ranking heatmap visualization
    fig_heatmap = plot_ranking_heatmap(
        df=df,
        figsize=(12, 8),
        title="Rankings Across Datasets"
    )
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
