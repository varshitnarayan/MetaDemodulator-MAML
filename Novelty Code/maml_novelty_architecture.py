import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. DATA GENERATOR
# ==========================================
def get_16qam_symbols():
    coords = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    return torch.tensor([complex(r, i) for r in coords for i in coords], dtype=torch.complex64)

def generate_device_data(num_samples, h=None, beta=None, snr_db=21, is_pilot=False):
    symbols = get_16qam_symbols()
    if h is None: h = (torch.randn(1) + 1j * torch.randn(1)) / math.sqrt(2)
    if beta is None: beta = torch.empty(1).uniform_(0.05, 0.15).item()
        
    if is_pilot and num_samples > 0:
        labels = torch.arange(num_samples) % 16
    else:
        labels = torch.randint(0, 16, (num_samples,))
    
    s = symbols[labels]
    s_mag = torch.abs(s)
    x = ((4.0 * s_mag) / (1.0 + beta * (s_mag ** 2))) * torch.exp(1j * torch.angle(s))
    
    noise_var = torch.mean(torch.abs(symbols)**2) / (10 ** (snr_db / 10))
    noise = (torch.randn(num_samples) + 1j * torch.randn(num_samples)) * math.sqrt(noise_var / 2)
    
    y = h * x + noise
    
    # 4D Features for the Enhanced Model
    features_4d = torch.stack((y.real, y.imag, torch.abs(y), torch.angle(y)), dim=1).float()
    return features_4d, labels, h, beta

# ==========================================
# 2. ARCHITECTURES
# ==========================================

# --- EXACT REPRODUCTION BASELINE (10 Neurons, 3 Layers, 2D Input) ---
def init_baseline_weights():
    he = lambda out_f, in_f: (torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)).requires_grad_(True)
    return {
        'w1': he(10, 2),  'b1': torch.zeros(10, requires_grad=True),    
        'w2': he(10, 10), 'b2': torch.zeros(10, requires_grad=True),
        'w3': he(16, 10), 'b3': torch.zeros(16, requires_grad=True)
    }

def forward_baseline(x_4d, weights):
    x_2d = x_4d[:, :2] # Baseline strictly only uses Real/Imaginary
    h1 = F.relu(F.linear(x_2d, weights['w1'], weights['b1']))
    h2 = F.relu(F.linear(h1, weights['w2'], weights['b2']))
    return F.linear(h2, weights['w3'], weights['b3'])

# --- ENHANCED NOVELTY (16 Neurons, 4 Layer ResNet, 4D Input) ---
def init_enhanced_weights():
    he = lambda out_f, in_f: (torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)).requires_grad_(True)
    return {
        'w1': he(16, 4),  'b1': torch.zeros(16, requires_grad=True),    
        'w2': he(16, 16), 'b2': torch.zeros(16, requires_grad=True),
        'w3': he(16, 16), 'b3': torch.zeros(16, requires_grad=True),
        'w4': he(16, 16), 'b4': torch.zeros(16, requires_grad=True)
    }

def forward_enhanced(x_4d, weights):
    h1 = F.relu(F.linear(x_4d, weights['w1'], weights['b1']))
    h2 = F.relu(F.linear(h1, weights['w2'], weights['b2']))
    h3_pre = F.linear(h2, weights['w3'], weights['b3'])
    
    # True ResNet Skip Connection
    h3 = F.relu(h3_pre + h1) 
    
    return F.linear(h3, weights['w4'], weights['b4'])

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate_model(weights, forward_fn, P, payload_size=10000, lr=0.01, adapt_steps=100):
    y_pilots, labels_pilots, device_h, device_beta = generate_device_data(P, is_pilot=True)
    adapted_weights = {k: v.clone() for k, v in weights.items()}
    
    if P > 0:
        for _ in range(adapt_steps):
            logits = forward_fn(y_pilots, adapted_weights)
            grads = torch.autograd.grad(F.cross_entropy(logits, labels_pilots), adapted_weights.values())
            adapted_weights = {name: w - lr * g for ((name, w), g) in zip(adapted_weights.items(), grads)}
        
    y_payload, labels_payload, _, _ = generate_device_data(payload_size, h=device_h, beta=device_beta)
    with torch.no_grad():
        preds = torch.argmax(forward_fn(y_payload, adapted_weights), dim=1)
    return torch.sum(preds != labels_payload).item() / payload_size

# ==========================================
# 4. PLOTTING THE COMPARISON
# ==========================================
def generate_comparison_figure(base_w, nov_w, joint_w, fixed_w):
    print("\nGenerating the Official Baseline vs. Enhanced Graph...")
    p_values = [0, 2, 4, 8, 12, 14, 16]
    ser_base, ser_nov, ser_joint, ser_fixed = [], [], [], []
    num_test_devices = 50 
    
    for P in p_values:
        # Both models get 100 steps to ensure a fair, identical fight
        avg_nov = sum(evaluate_model(nov_w, forward_enhanced, P, adapt_steps=100) for _ in range(num_test_devices)) / num_test_devices
        avg_base = sum(evaluate_model(base_w, forward_baseline, P, adapt_steps=100) for _ in range(num_test_devices)) / num_test_devices
        
        # Traditional Baselines
        avg_joint = sum(evaluate_model(joint_w, forward_baseline, P, adapt_steps=500) for _ in range(num_test_devices)) / num_test_devices
        avg_fixed = sum(evaluate_model(fixed_w, forward_baseline, P, adapt_steps=500) for _ in range(num_test_devices)) / num_test_devices
            
        ser_nov.append(avg_nov); ser_base.append(avg_base); ser_joint.append(avg_joint); ser_fixed.append(avg_fixed)
        print(f"P={P:2d} | Original MAML SER: {avg_base:.4f} | Enhanced 4D ResNet SER: {avg_nov:.4f}")
        
    plt.figure(figsize=(9, 7))
    plt.plot(p_values, ser_fixed, label='Fixed Initialization', linestyle='--', color='green')
    plt.plot(p_values, ser_joint, label='Joint Training', linestyle='-', marker='s', markerfacecolor='none', color='blue')
    plt.plot(p_values, ser_base, label='Original MAML (2D MLP)', linestyle='-', marker='o', markerfacecolor='none', color='red', alpha=0.6)
    plt.plot(p_values, ser_nov, label='Proposed MAML (4D ResNet)', linestyle='-', marker='*', markersize=10, color='darkorange')
    
    plt.yscale('log') 
    plt.xlabel('Number of pilots for meta-test device (P)', fontsize=12)
    plt.ylabel('Probability of symbol error', fontsize=12)
    plt.title('Demodulation Performance: Original vs. Proposed Architecture', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/Figure_Novelty_Architecture.png', dpi=300)
    plt.show()

# ==========================================
# 5. THE SIMULTANEOUS TRAINING LOOP
# ==========================================
def run_comparative_scenario():
    K = 20
    N_tr = 16 
    N_te = 16
    inner_lr = 0.01
    meta_lr = 0.001
    meta_epochs = 5000  
    
    base_weights = init_baseline_weights()
    nov_weights = init_enhanced_weights()
    joint_weights = init_baseline_weights()
    fixed_weights = init_baseline_weights()
    
    base_opt = torch.optim.Adam(base_weights.values(), lr=meta_lr) 
    nov_opt = torch.optim.Adam(nov_weights.values(), lr=meta_lr) 
    joint_opt = torch.optim.Adam(joint_weights.values(), lr=meta_lr)
    
    print("Starting Training: Exact Baseline vs. Proposed 4D ResNet...")
    fixed_devices = [generate_device_data(1)[2:4] for _ in range(K)]
    
    for epoch in range(meta_epochs):
        base_loss = 0.0; nov_loss = 0.0; joint_loss = 0.0
        for h_k, beta_k in fixed_devices:
            y_j, labels_j, _, _ = generate_device_data(N_tr + N_te, h=h_k, beta=beta_k, is_pilot=True)
            y_tr, labels_tr, _, _ = generate_device_data(N_tr, h=h_k, beta=beta_k, is_pilot=True)
            y_te, labels_te, _, _ = generate_device_data(N_te, h=h_k, beta=beta_k, is_pilot=True)
            
            # 1. Joint Baseline
            joint_loss += F.cross_entropy(forward_baseline(y_j, joint_weights), labels_j)
            
            # 2. Original Baseline MAML (Trains with 1 Step)
            loss_tr_base = F.cross_entropy(forward_baseline(y_tr, base_weights), labels_tr)
            grads_base = torch.autograd.grad(loss_tr_base, base_weights.values(), create_graph=True)
            fast_w_base = {name: w - inner_lr * g for ((name, w), g) in zip(base_weights.items(), grads_base)}
            base_loss += F.cross_entropy(forward_baseline(y_te, fast_w_base), labels_te)
            
            # 3. Proposed MAML (Trains with 1 Step, identically to baseline)
            loss_tr_nov = F.cross_entropy(forward_enhanced(y_tr, nov_weights), labels_tr)
            grads_nov = torch.autograd.grad(loss_tr_nov, nov_weights.values(), create_graph=True)
            fast_w_nov = {name: w - inner_lr * g for ((name, w), g) in zip(nov_weights.items(), grads_nov)}
            nov_loss += F.cross_entropy(forward_enhanced(y_te, fast_w_nov), labels_te)
            
        joint_opt.zero_grad(); (joint_loss / K).backward(); joint_opt.step()
        base_opt.zero_grad(); (base_loss / K).backward(); base_opt.step()
        nov_opt.zero_grad(); (nov_loss / K).backward(); nov_opt.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1:4d} | Baseline Loss: {(base_loss/K):.4f} | Proposed ResNet Loss: {(nov_loss/K):.4f}")

    print("Training Complete!")
    generate_comparison_figure(base_weights, nov_weights, joint_weights, fixed_weights)

if __name__ == "__main__":
    run_comparative_scenario()