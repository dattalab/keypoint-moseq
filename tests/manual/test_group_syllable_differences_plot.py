"""
Manual visual tests for group syllable differences plotting.
Run this script to generate test plots for visual inspection.

Usage:
    python tests/manual/test_group_syllable_differences_plot.py
"""

import numpy as np
from keypoint_moseq.view.jupyter_display import _group_syllable_differences_plot

def test_simple_2group():
    """Test case 1: Simple 2-group comparison with 3 syllables"""
    
    # Test data: 3 syllables, 2 groups
    centers = np.array([
        [0.8, 1.2],  # syllable 0
        [1.5, 0.9],  # syllable 1  
        [0.6, 0.7]   # syllable 2
    ])
    errors = np.array([
        [0.1, 0.15],
        [0.2, 0.1], 
        [0.08, 0.12]
    ])
    significant = [
        [(0, 1)],  # syllable 0: groups 0 vs 1 significant
        [],        # syllable 1: no significance
        []         # syllable 2: no significance  
    ]
    group_labels = ['Control', 'Treatment']
    syllables = ['0', '1', '2']
    y_axis_label = 'Mean Usage Rate'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_simple_2group.png', dpi=150, bbox_inches='tight')
    print("Saved: test_simple_2group.png")


def test_multi_group():
    """Test case 2: Multi-group comparison with mixed significance patterns"""
    
    # Test data: 4 syllables, 4 groups
    centers = np.array([
        [1.2, 0.8, 1.5, 0.9],  # syllable 0
        [0.7, 1.1, 0.6, 1.3],  # syllable 1
        [1.0, 1.4, 0.8, 1.1],  # syllable 2
        [0.9, 0.5, 1.2, 0.8]   # syllable 3
    ])
    errors = np.array([
        [0.12, 0.08, 0.18, 0.10],
        [0.09, 0.14, 0.07, 0.16],
        [0.11, 0.20, 0.09, 0.13],
        [0.10, 0.06, 0.15, 0.09]
    ])
    significant = [
        [(0, 2), (1, 3)],  # syllable 0: two comparisons
        [(0, 1)],          # syllable 1: one comparison
        [],                # syllable 2: no significance
        [(1, 2), (2, 3), (0, 3)]  # syllable 3: three comparisons (test stacking)
    ]
    group_labels = ['Wild-type', 'Mutant A', 'Mutant B', 'Rescue']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Frequency (Hz)'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_multi_group.png', dpi=150, bbox_inches='tight')
    print("Saved: test_multi_group.png")


def test_many_syllables():
    """Test case 3: Many syllables with sparse significance"""
    
    # Test data: 57 syllables, 3 groups
    np.random.seed(42)  # For reproducible test data
    centers = np.random.randn(57, 3) * 0.3 + 1.0  # centered around 1.0
    errors = np.abs(np.random.randn(57, 3)) * 0.1 + 0.05  # small positive errors
    
    # Sparse significance - about 10% of syllables significant
    significant = [[] for _ in range(57)]  # Initialize all empty
    significant[3] = [(0, 1)]
    significant[12] = [(1, 2)]
    significant[23] = [(0, 2)]
    significant[34] = [(0, 1)]
    significant[45] = [(1, 2)]
    significant[52] = [(0, 2)]
    
    group_labels = ['Group A', 'Group B', 'Group C']
    syllables = [str(i) for i in range(57)]
    y_axis_label = 'Normalized Expression'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_many_syllables.png', dpi=150, bbox_inches='tight')
    print("Saved: test_many_syllables.png")


def test_single_syllable():
    """Test case 4: Single syllable with multiple groups and stacked significance"""
    
    # Test data: 1 syllable, 4 groups
    centers = np.array([[1.2, 0.8, 1.5, 0.6]])  # shape: (1, 4)
    errors = np.array([[0.15, 0.12, 0.18, 0.10]])
    
    # Multiple significant comparisons to test vertical stacking
    significant = [
        [(0, 1), (0, 2), (1, 3), (2, 3)]  # 4 comparisons for stacking test
    ]
    group_labels = ['Baseline', 'Low Dose', 'High Dose', 'Recovery']
    syllables = ['0']
    y_axis_label = 'Response Magnitude'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_single_syllable.png', dpi=150, bbox_inches='tight')
    print("Saved: test_single_syllable.png")


def test_no_significance():
    """Test case 5: No significance - all empty lists"""
    
    # Test data: 4 syllables, 3 groups, no significance
    centers = np.array([
        [1.1, 0.9, 1.3],
        [0.8, 1.2, 0.7],
        [1.4, 1.0, 1.1],
        [0.6, 0.9, 1.2]
    ])
    errors = np.array([
        [0.12, 0.10, 0.15],
        [0.08, 0.14, 0.09],
        [0.16, 0.11, 0.13],
        [0.07, 0.10, 0.14]
    ])
    significant = [[], [], [], []]  # No significance anywhere
    group_labels = ['Group X', 'Group Y', 'Group Z']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Activity Level'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_no_significance.png', dpi=150, bbox_inches='tight')
    print("Saved: test_no_significance.png")


def test_all_significant():
    """Test case 6: Every possible group comparison significant for every syllable"""
    
    # Test data: 3 syllables, 3 groups (3 possible pairwise comparisons per syllable)
    centers = np.array([
        [1.5, 0.8, 1.2],
        [0.9, 1.4, 0.7],
        [1.1, 0.6, 1.3]
    ])
    errors = np.array([
        [0.15, 0.08, 0.12],
        [0.09, 0.14, 0.07],
        [0.11, 0.06, 0.13]
    ])
    # All possible comparisons for 3 groups: (0,1), (0,2), (1,2)
    significant = [
        [(0, 1), (0, 2), (1, 2)],  # syllable 0: all comparisons
        [(0, 1), (0, 2), (1, 2)],  # syllable 1: all comparisons
        [(0, 1), (0, 2), (1, 2)]   # syllable 2: all comparisons
    ]
    group_labels = ['Control', 'Treatment A', 'Treatment B']
    syllables = ['0', '1', '2']
    y_axis_label = 'Response Score'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_all_significant.png', dpi=150, bbox_inches='tight')
    print("Saved: test_all_significant.png")


def test_large_error_bars():
    """Test case 7: Large error bars (50%+ of center values)"""
    
    # Test data: 3 syllables, 3 groups with very large error bars
    centers = np.array([
        [2.0, 1.5, 2.2],
        [1.8, 2.1, 1.4],
        [1.6, 1.9, 2.0]
    ])
    # Error bars that are 50-80% of center values
    errors = np.array([
        [1.0, 0.9, 1.3],  # 50%, 60%, 59% of centers
        [1.1, 1.2, 0.8],  # 61%, 57%, 57% of centers  
        [0.9, 1.4, 1.1]   # 56%, 74%, 55% of centers
    ])
    significant = [
        [(0, 1)],  # syllable 0
        [],        # syllable 1
        [(1, 2)]   # syllable 2
    ]
    group_labels = ['Condition 1', 'Condition 2', 'Condition 3']
    syllables = ['0', '1', '2']
    y_axis_label = 'Variable Measure'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_large_error_bars.png', dpi=150, bbox_inches='tight')
    print("Saved: test_large_error_bars.png")


def test_many_stacked_bars():
    """Test case 8: One syllable with many stacked significance bars"""
    
    # Test data: 2 syllables, 6 groups to create many possible comparisons
    centers = np.array([
        [1.0, 1.3, 0.8, 1.5, 0.7, 1.2],  # syllable 0
        [0.9, 1.1, 1.0, 0.8, 1.2, 0.95]  # syllable 1
    ])
    errors = np.array([
        [0.1, 0.13, 0.08, 0.15, 0.07, 0.12],
        [0.09, 0.11, 0.10, 0.08, 0.12, 0.095]
    ])
    significant = [
        [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (0, 5)],  # 6 comparisons - heavy stacking
        [(1, 2)]  # syllable 1: just one for contrast
    ]
    group_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    syllables = ['0', '1']
    y_axis_label = 'Signal Intensity'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_many_stacked_bars.png', dpi=150, bbox_inches='tight')
    print("Saved: test_many_stacked_bars.png")


def test_minimal_spacing():
    """Test case 9: Groups with very similar values (test visual separation)"""
    
    # Test data: 4 syllables, 4 groups with very similar values
    centers = np.array([
        [1.000, 1.002, 1.001, 1.003],  # Very close values
        [0.998, 1.001, 0.999, 1.000],
        [1.001, 0.999, 1.002, 0.998],
        [1.002, 1.000, 0.999, 1.001]
    ])
    errors = np.array([
        [0.0005, 0.0008, 0.0006, 0.0007],  # Small errors relative to differences
        [0.0006, 0.0009, 0.0005, 0.0008],
        [0.0007, 0.0005, 0.0008, 0.0006],
        [0.0008, 0.0006, 0.0005, 0.0009]
    ])
    significant = [
        [(0, 3)],  # syllable 0
        [],        # syllable 1
        [(1, 3)],  # syllable 2
        [(0, 2)]   # syllable 3
    ]
    group_labels = ['Type I', 'Type II', 'Type III', 'Type IV']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Precise Measurement'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_minimal_spacing.png', dpi=150, bbox_inches='tight')
    print("Saved: test_minimal_spacing.png")


def test_single_group():
    """Test case 10: Single group - no comparisons possible"""
    
    # Test data: 4 syllables, 1 group only
    centers = np.array([
        [1.2],  # syllable 0
        [0.8],  # syllable 1
        [1.5],  # syllable 2
        [0.9]   # syllable 3
    ])
    errors = np.array([
        [0.15],
        [0.12],
        [0.18],
        [0.10]
    ])
    significant = [[], [], [], []]  # No significance possible with 1 group
    group_labels = ['Single Group']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Activity Level'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_single_group.png', dpi=150, bbox_inches='tight')
    print("Saved: test_single_group.png")


def run_all_visual_tests():
    """Run all visual test cases and generate plots for inspection."""
    print("Running all visual tests for group syllable differences plotting...")
    print()
    
    test_simple_2group()
    test_multi_group() 
    test_many_syllables()
    test_single_syllable()
    test_no_significance()
    test_all_significant()
    test_large_error_bars()
    test_many_stacked_bars()
    test_minimal_spacing()
    test_single_group()
    
    print()
    print("All visual tests completed! Check the generated PNG files for visual inspection.")


if __name__ == "__main__":
    run_all_visual_tests()

