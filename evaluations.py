### t-SNE Graphing 

import numpy as np
import matplotlib.pyplot as plt #type: ignore
from tqdm import tqdm #type: ignore
from sklearn.manifold import TSNE #type: ignore
import seaborn as sns #type: ignore


def visualize_embeddings_tsne(nodes, embeddings, output_file="lab6-embeddings_tsne.png", 
                        sample_size=128, annotate=True, label_map=None):
    """Create t-SNE visualization of embeddings.
    Args:
        nodes: List of node/word labels
        embeddings: Array of embedding vectors
        output_file: Path to save the t-SNE plot
        sample_size: Number of top embeddings to visualize
        annotate: Whether to label points with node names
    
    Saves:
        PNG file to output_file with t-SNE scatter plot of embeddings
    """
    if type(embeddings) != np.ndarray:
        embeddings = embeddings.cpu().numpy()
     
    n_samples = min(sample_size, len(nodes))
    selected_embeddings = embeddings[:n_samples]
    
    selected_nodes = nodes[:n_samples]
    if label_map:
        selected_nodes = [label_map.get(node, str(node)) for node in nodes[:n_samples]]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(14, 14))
    sns.scatterplot(
        x=projection[:, 0], 
        y=projection[:, 1], 
        hue=selected_nodes,  # This colors the dots by your labels (e.g., Emotion)
        palette="muted",
        s=60, 
        alpha=0.7,
        edgecolor='w'
    )
    
    if annotate:
        for i, word in enumerate(selected_nodes):
            plt.annotate(word, (projection[i, 0], projection[i, 1]), fontsize=9, alpha=0.8)
            
    plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"VGGish Latent Space t-SNE ({n_samples} Samples)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE to {output_file}")
    plt.show()
    plt.close()
    

def create_visualizations(sim_matrix, labels, class_words, label_to_word, images=None, names=None, predictions=None):
    """Create all visualizations in one coordinated function."""
    # OOD analysis if provided
    if images and names and predictions:
        print(f"\n📸 Creating OOD visualization for {len(images)} images...")
        n_imgs = len(images)
        n_cols = min(4, n_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 7*n_rows))
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        for i, (img, name, pred) in enumerate(zip(images, names, predictions)):
            axes[i].imshow(img); axes[i].axis('off')
            pred_text = f"{name.upper()}\n\nTop matches:\n"
            for rank, (word, sim) in enumerate(zip(pred['words'][:5], pred['sims'][:5]), 1):
                pred_text += f"{rank}. {word} ({sim:.3f})\n"
            axes[i].set_title(pred_text, fontsize=11, ha='center', color='darkblue', fontweight='bold', pad=12)
        
        for j in range(i+1, len(axes)): 
            axes[j].axis('off'); axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('ood_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    else:
        print("\n📊 Creating confusion matrix...")
        n_classes = len(class_words)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i, label in enumerate(labels):
            if (word := label_to_word[label]) in class_words:
                true_idx = class_words.index(word)
                pred_idx = np.argmax(sim_matrix[i])
                conf_matrix[true_idx, pred_idx] += 1
        
        conf_matrix = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, xticklabels=class_words, yticklabels=class_words,
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Probability'}, square=True)
        ax.set_xlabel('Predicted Class'); ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (All Classes)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
