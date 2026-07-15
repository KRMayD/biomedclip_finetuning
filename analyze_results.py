#!/usr/bin/env python3
"""
Results Analysis Script
=======================
Analyzes all experimental results and finds best configurations.
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    return parser.parse_args()


def load_results(results_file):
    """Load results from CSV"""
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    return df


def analyze_by_loss(df):
    """Analyze results grouped by loss function"""
    print("\n" + "="*80)
    print("Analysis by Loss Function")
    print("="*80)
    
    loss_stats = df.groupby('Loss').agg({
        'Accuracy': ['mean', 'std', 'max'],
        'F1_Score': ['mean', 'std', 'max'],
        'AUC_ROC': ['mean', 'std', 'max']
    }).round(4)
    
    print(loss_stats)
    print()
    
    # Best configuration for each loss
    print("\nBest Configuration per Loss Function:")
    print("-" * 80)
    
    best_configs = []
    for loss in df['Loss'].unique():
        loss_df = df[df['Loss'] == loss]
        best_idx = loss_df['F1_Score'].idxmax()
        best_row = loss_df.loc[best_idx]
        
        print(f"\n{loss.upper()}:")
        print(f"  Batch Size: {best_row['Batch_Size']}, Epoch: {best_row['Epoch']}")
        print(f"  Accuracy: {best_row['Accuracy']:.4f}, F1: {best_row['F1_Score']:.4f}, AUC: {best_row['AUC_ROC']:.4f}")
        
        best_configs.append({
            'Loss': loss,
            'Batch_Size': best_row['Batch_Size'],
            'Epoch': best_row['Epoch'],
            'Accuracy': best_row['Accuracy'],
            'F1_Score': best_row['F1_Score'],
            'AUC_ROC': best_row['AUC_ROC']
        })
    
    return pd.DataFrame(best_configs)


def analyze_by_batch_size(df):
    """Analyze results grouped by batch size"""
    print("\n" + "="*80)
    print("Analysis by Batch Size")
    print("="*80)
    
    batch_stats = df.groupby('Batch_Size').agg({
        'Accuracy': ['mean', 'std', 'max'],
        'F1_Score': ['mean', 'std', 'max'],
        'AUC_ROC': ['mean', 'std', 'max']
    }).round(4)
    
    print(batch_stats)
    print()


def analyze_by_epoch(df):
    """Analyze results grouped by epoch"""
    print("\n" + "="*80)
    print("Analysis by Epoch")
    print("="*80)
    
    epoch_stats = df.groupby('Epoch').agg({
        'Accuracy': ['mean', 'std', 'max'],
        'F1_Score': ['mean', 'std', 'max'],
        'AUC_ROC': ['mean', 'std', 'max']
    }).round(4)
    
    print(epoch_stats)
    print()


def find_overall_best(df):
    """Find overall best configurations"""
    print("\n" + "="*80)
    print("Overall Best Configurations")
    print("="*80)
    
    # Best by F1 Score
    best_f1_idx = df['F1_Score'].idxmax()
    best_f1 = df.loc[best_f1_idx]
    
    print("\nBest F1 Score:")
    print(f"  Experiment: {best_f1['Experiment']}")
    print(f"  Loss: {best_f1['Loss']}, Batch: {best_f1['Batch_Size']}, Epoch: {best_f1['Epoch']}")
    print(f"  Accuracy: {best_f1['Accuracy']:.4f}, F1: {best_f1['F1_Score']:.4f}, AUC: {best_f1['AUC_ROC']:.4f}")
    
    # Best by Accuracy
    best_acc_idx = df['Accuracy'].idxmax()
    best_acc = df.loc[best_acc_idx]
    
    print("\nBest Accuracy:")
    print(f"  Experiment: {best_acc['Experiment']}")
    print(f"  Loss: {best_acc['Loss']}, Batch: {best_acc['Batch_Size']}, Epoch: {best_acc['Epoch']}")
    print(f"  Accuracy: {best_acc['Accuracy']:.4f}, F1: {best_acc['F1_Score']:.4f}, AUC: {best_acc['AUC_ROC']:.4f}")
    
    # Best by AUC
    best_auc_idx = df['AUC_ROC'].idxmax()
    best_auc = df.loc[best_auc_idx]
    
    print("\nBest AUC-ROC:")
    print(f"  Experiment: {best_auc['Experiment']}")
    print(f"  Loss: {best_auc['Loss']}, Batch: {best_auc['Batch_Size']}, Epoch: {best_auc['Epoch']}")
    print(f"  Accuracy: {best_auc['Accuracy']:.4f}, F1: {best_auc['F1_Score']:.4f}, AUC: {best_auc['AUC_ROC']:.4f}")
    
    return {
        'best_f1': best_f1,
        'best_accuracy': best_acc,
        'best_auc': best_auc
    }


def create_comparison_table(df):
    """Create detailed comparison table"""
    print("\n" + "="*80)
    print("Detailed Comparison Table (Top 10 by F1 Score)")
    print("="*80)
    
    top_10 = df.nlargest(10, 'F1_Score')[['Experiment', 'Loss', 'Batch_Size', 'Epoch', 
                                            'Accuracy', 'F1_Score', 'AUC_ROC']]
    print(top_10.to_string(index=False))
    print()
    
    return top_10


def save_summary(df, best_configs_df, best_overall, results_dir):
    """Save analysis summary to files"""
    results_dir = Path(results_dir)
    
    # Save best configurations
    best_file = results_dir / 'best_results.csv'
    best_configs_df.to_csv(best_file, index=False)
    print(f"✓ Best configurations saved to: {best_file}")
    
    # Save summary text
    summary_file = results_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BiomedCLIP Fine-tuning Results Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL BEST CONFIGURATIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Best F1 Score:\n")
        best_f1 = best_overall['best_f1']
        f.write(f"  Experiment: {best_f1['Experiment']}\n")
        f.write(f"  Loss: {best_f1['Loss']}, Batch: {best_f1['Batch_Size']}, Epoch: {best_f1['Epoch']}\n")
        f.write(f"  Accuracy: {best_f1['Accuracy']:.4f}, F1: {best_f1['F1_Score']:.4f}, AUC: {best_f1['AUC_ROC']:.4f}\n\n")
        
        f.write("Best Accuracy:\n")
        best_acc = best_overall['best_accuracy']
        f.write(f"  Experiment: {best_acc['Experiment']}\n")
        f.write(f"  Loss: {best_acc['Loss']}, Batch: {best_acc['Batch_Size']}, Epoch: {best_acc['Epoch']}\n")
        f.write(f"  Accuracy: {best_acc['Accuracy']:.4f}, F1: {best_acc['F1_Score']:.4f}, AUC: {best_acc['AUC_ROC']:.4f}\n\n")
        
        f.write("Best AUC-ROC:\n")
        best_auc = best_overall['best_auc']
        f.write(f"  Experiment: {best_auc['Experiment']}\n")
        f.write(f"  Loss: {best_auc['Loss']}, Batch: {best_auc['Batch_Size']}, Epoch: {best_auc['Epoch']}\n")
        f.write(f"  Accuracy: {best_auc['Accuracy']:.4f}, F1: {best_auc['F1_Score']:.4f}, AUC: {best_auc['AUC_ROC']:.4f}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST CONFIGURATION PER LOSS FUNCTION\n")
        f.write("="*80 + "\n\n")
        
        for _, row in best_configs_df.iterrows():
            f.write(f"{row['Loss'].upper()}:\n")
            f.write(f"  Batch Size: {row['Batch_Size']}, Epoch: {row['Epoch']}\n")
            f.write(f"  Accuracy: {row['Accuracy']:.4f}, F1: {row['F1_Score']:.4f}, AUC: {row['AUC_ROC']:.4f}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICS BY LOSS FUNCTION\n")
        f.write("="*80 + "\n\n")
        
        loss_stats = df.groupby('Loss').agg({
            'Accuracy': ['mean', 'std', 'max'],
            'F1_Score': ['mean', 'std', 'max'],
            'AUC_ROC': ['mean', 'std', 'max']
        }).round(4)
        
        f.write(loss_stats.to_string())
        f.write("\n")
    
    print(f"✓ Summary saved to: {summary_file}")


def main():
    args = parse_args()
    
    results_file = Path(args.results_dir) / 'all_results.csv'
    
    # Load results
    df = load_results(results_file)
    if df is None or len(df) == 0:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(df)} experiment results")
    print(f"Losses: {sorted(df['Loss'].unique())}")
    print(f"Batch sizes: {sorted(df['Batch_Size'].unique())}")
    print(f"Epochs: {sorted(df['Epoch'].unique())}")
    
    # Analyze results
    best_configs_df = analyze_by_loss(df)
    analyze_by_batch_size(df)
    analyze_by_epoch(df)
    best_overall = find_overall_best(df)
    create_comparison_table(df)
    
    # Save summary
    save_summary(df, best_configs_df, best_overall, args.results_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
