import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. DATA GENERATOR
# ==========================================
def generate_device_data(num_samples, h=None, snr_db=15, is_pilot=False):
    symbols = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    if h is None:
        h = 1.0 if np.random.rand() > 0.5 else -1.0
    if is_pilot and num_samples > 0:
        labels = torch.arange(num_samples) % 4
    else:
        labels = torch.randint(0, 4, (num_samples,))
    s = symbols[labels]
    
    snr_linear = 10 ** (snr_db / 10)
    E_s2 = torch.mean(symbols**2)
    noise_variance = E_s2 / snr_linear
    noise = torch.randn(num_samples) * torch.sqrt(noise_variance)
    
    y = h * s + noise
    y_2d = torch.stack((y, torch.zeros_like(y)), dim=1)
    return y_2d, labels, h

# ==========================================
# 2. FUNCTIONAL NEURAL NETWORK
# ==========================================
def initialize_meta_weights():
    return {
        'w1': (torch.randn(30, 2) * 0.1).requires_grad_(True), 
        'b1': torch.zeros(30, requires_grad=True),    
        'w2': (torch.randn(4, 30) * 0.1).requires_grad_(True), 
        'b2': torch.zeros(4, requires_grad=True)      
    }

def functional_forward(x, weights):
    hidden = F.linear(x, weights['w1'], weights['b1'])
    hidden = torch.tanh(hidden)
    return F.linear(hidden, weights['w2'], weights['b2'])

# ==========================================
# 3. EVALUATION FUNCTION (Shared)
# ==========================================
def evaluate_model(weights, P, snr_db=15, payload_size=10000, lr=0.1, adapt_steps=1):
    y_pilots, labels_pilots, device_h = generate_device_data(P, snr_db=snr_db, is_pilot=True)
    adapted_weights = {k: v.clone() for k, v in weights.items()}
    
    if P > 0:
        for _ in range(adapt_steps):
            logits_pilots = functional_forward(y_pilots, adapted_weights)
            loss_pilots = F.cross_entropy(logits_pilots, labels_pilots)
            grads = torch.autograd.grad(loss_pilots, adapted_weights.values())
            adapted_weights = {name: w - lr * g for ((name, w), g) in zip(adapted_weights.items(), grads)}
            
    y_payload, labels_payload, _ = generate_device_data(payload_size, h=device_h, snr_db=snr_db)
    with torch.no_grad():
        preds = torch.argmax(functional_forward(y_payload, adapted_weights), dim=1)
        ser = torch.sum(preds != labels_payload).item() / payload_size
    return ser

# ==========================================
# 4. PLOTTING FIGURES 4 & 5
# ==========================================
def generate_figure_4(maml_weights, joint_weights, fixed_weights):
    print("\nGenerating Figure 4 (Evaluating SER vs Pilots...)")
    p_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ser_maml, ser_joint, ser_fixed = [], [], []
    num_test_devices = 20 
    
    for P in p_values:
        avg_m = sum(evaluate_model(maml_weights, P, adapt_steps=1) for _ in range(num_test_devices)) / num_test_devices
        avg_j = sum(evaluate_model(joint_weights, P, adapt_steps=100) for _ in range(num_test_devices)) / num_test_devices
        avg_f = sum(evaluate_model(fixed_weights, P, adapt_steps=100) for _ in range(num_test_devices)) / num_test_devices
        
        ser_maml.append(avg_m); ser_joint.append(avg_j); ser_fixed.append(avg_f)
        print(f"P={P} | Fixed: {avg_f:.4f} | Joint: {avg_j:.4f} | MAML: {avg_m:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, ser_fixed, label='fixed initialization', linestyle='--', color='green')
    plt.plot(p_values, ser_joint, label='joint training', linestyle='-', marker='s', markerfacecolor='none', color='blue')
    plt.plot(p_values, ser_maml, label='MAML', linestyle='-', marker='o', markerfacecolor='none', color='red')
    
    plt.yscale('log')
    plt.xlabel('number of pilots for meta-test device (P)', fontsize=12)
    plt.ylabel('probability of symbol error', fontsize=12)
    plt.title('Figure 4: Demodulation Performance (Toy Example)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/Figure_4.png', dpi=300)
    plt.show()

def generate_figure_5(maml_weights, lr=0.1):
    print("\nGenerating Figure 5 (Probability Distributions...)")
    
    device_h = -1.0
    y_pilots, labels_pilots, _ = generate_device_data(6, h=device_h, is_pilot=True)
    
    adapted_weights = {k: v.clone() for k, v in maml_weights.items()}
    logits_pilots = functional_forward(y_pilots, adapted_weights)
    grads = torch.autograd.grad(F.cross_entropy(logits_pilots, labels_pilots), adapted_weights.values())
    adapted_weights = {name: w - lr * g for ((name, w), g) in zip(adapted_weights.items(), grads)}
    
    y_real = torch.linspace(-5, 5, 200)
    y_2d = torch.stack((y_real, torch.zeros_like(y_real)), dim=1)
    
    with torch.no_grad():
        probs_maml = F.softmax(functional_forward(y_2d, maml_weights), dim=1).numpy()
        probs_adapt = F.softmax(functional_forward(y_2d, adapted_weights), dim=1).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    colors = ['black', 'green', 'blue', 'purple']
    
    for i in range(4):
        ax1.plot(y_real.numpy(), probs_maml[:, i], color=colors[i])
        ax2.plot(y_real.numpy(), probs_adapt[:, i], color=colors[i])

    ax1.text(-3.8, 0.45, 's = 3', color='purple', fontsize=12)
    ax1.text(-1.5, 0.45, 's = 1', color='blue', fontsize=12)
    ax1.text(0.5, 0.45, 's = -1', color='green', fontsize=12)
    ax1.text(3.0, 0.45, 's = -3', color='black', fontsize=12)

    ax2.text(-3.8, 0.85, 's = 3', color='purple', fontsize=12)
    ax2.text(-1.5, 0.85, 's = 1', color='blue', fontsize=12)
    ax2.text(0.5, 0.85, 's = -1', color='green', fontsize=12)
    ax2.text(3.0, 0.85, 's = -3', color='black', fontsize=12)

    ax1.set_ylabel('probability $p_\\theta(s|y)$', fontsize=12)
    ax1.set_xlabel('received signal ($y$)', fontsize=12)
    ax1.set_xlim([-4.5, 4.5]); ax1.set_ylim([0, 1.05])
    
    ax2.set_ylabel('probability $p(s|y, \\phi_T)$', fontsize=12)
    ax2.set_xlabel('received signal ($y$)', fontsize=12)
    ax2.set_xlim([-4.5, 4.5]); ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/Figure_5.png', dpi=300)
    plt.show()

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
def run_toy_example():
    K = 20; N_tr = 4; N_te = 4; inner_lr = 0.1; meta_lr = 0.025; meta_epochs = 5000  
    
    maml_weights = initialize_meta_weights()
    joint_weights = initialize_meta_weights() 
    fixed_weights = initialize_meta_weights()
    
    maml_optimizer = torch.optim.SGD(maml_weights.values(), lr=meta_lr)
    joint_optimizer = torch.optim.SGD(joint_weights.values(), lr=meta_lr)
    
    print("Starting Training (MAML & Joint)...")
    for epoch in range(meta_epochs):
        maml_loss = 0.0; joint_loss = 0.0
        for k in range(K):
            current_h = 1.0 if k < (K // 2) else -1.0
            
            y_j, labels_j, _ = generate_device_data(N_tr + N_te, h=current_h, is_pilot=True)
            joint_loss += F.cross_entropy(functional_forward(y_j, joint_weights), labels_j)
            
            y_tr, labels_tr, _ = generate_device_data(N_tr, h=current_h, is_pilot=True)
            loss_tr = F.cross_entropy(functional_forward(y_tr, maml_weights), labels_tr)
            grads = torch.autograd.grad(loss_tr, maml_weights.values(), create_graph=True)
            fast_w = {name: w - inner_lr * g for ((name, w), g) in zip(maml_weights.items(), grads)}
                
            y_te, labels_te, _ = generate_device_data(N_te, h=current_h, is_pilot=True)
            maml_loss += F.cross_entropy(functional_forward(y_te, fast_w), labels_te)
            
        joint_optimizer.zero_grad(); (joint_loss / K).backward(); joint_optimizer.step()
        maml_optimizer.zero_grad(); (maml_loss / K).backward(); maml_optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1} | MAML Loss: {(maml_loss/K):.4f} | Joint Loss: {(joint_loss/K):.4f}")

    print("Training Complete!")
    generate_figure_4(maml_weights, joint_weights, fixed_weights)
    generate_figure_5(maml_weights)

if __name__ == "__main__":
    run_toy_example()