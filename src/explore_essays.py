from datasets import load_dataset
import numpy as np

def explore_essays_dataset():
    # Load the dataset
    print("Loading essays-big5 dataset...")
    dataset = load_dataset("jingjietan/essays-big5")
    
    print("\nDataset Structure:")
    print("-" * 50)
    print(dataset)
    
    print("\nFeatures available:")
    print("-" * 50)
    for key in dataset['train'].features:
        print(key)
    
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Number of samples: {len(dataset['train'])}")
    
    # Calculate mean scores for each trait
    # O: Openness, C: Conscientiousness, E: Extraversion, A: Agreeableness, N: Neuroticism
    traits = ['O', 'C', 'E', 'A', 'N']
    trait_names = {
        'O': 'Openness',
        'C': 'Conscientiousness',
        'E': 'Extraversion',
        'A': 'Agreeableness',
        'N': 'Neuroticism'
    }
    
    print("\nMean Big-5 scores:")
    for trait in traits:
        mean_score = np.mean(dataset['train'][trait])
        std_score = np.std(dataset['train'][trait])
        print(f"{trait_names[trait]}: {mean_score:.3f} ± {std_score:.3f}")
    
    # Print a sample essay
    print("\nSample Essay:")
    print("-" * 50)
    idx = 0  # Get first essay
    print("Text excerpt (first 200 chars):", dataset['train']['text'][idx][:200], "...")
    print("\nPersonality Scores:")
    for trait in traits:
        print(f"{trait_names[trait]}: {dataset['train'][trait][idx]:.3f}")
    
    print("\nPersonality Types Distribution:")
    print("-" * 50)
    ptypes = dataset['train']['ptype']
    unique_types, counts = np.unique(ptypes, return_counts=True)
    for ptype, count in zip(unique_types, counts):
        percentage = (count / len(ptypes)) * 100
        print(f"{ptype}: {count} ({percentage:.1f}%)")

if __name__ == '__main__':
    explore_essays_dataset()
