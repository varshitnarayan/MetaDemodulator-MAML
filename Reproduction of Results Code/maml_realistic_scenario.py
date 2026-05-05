import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. DATA GENERATOR (Realistic Physics)
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
    return torch.stack((y.real, y.imag), dim=1).float(), labels, h, beta

# ==========================================
# 2. FUNCTIONAL NEURAL NETWORK
# ==========================================
def initialize_meta_weights():
    he = lambda out_f, in_f: (torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)).requires_grad_(True)
    return {
        'w1': he(10, 2), 'b1': torch.zeros(10, requires_grad=True),    
        'w2': he(10, 10), 'b2': torch.zeros(10, requires_grad=True),
        'w3': he(16, 10), 'b3': torch.zeros(16, requires_grad=True)
    }

def functional_forward(x, weights):
    h1 = F.relu(F.linear(x, weights['w1'], weights['b1']))
    h2 = F.relu(F.linear(h1, weights['w2'], weights['b2']))
    return F.linear(h2, weights['w3'], weights['b3'])

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate_model(weights, P, payload_size=10000, lr=0.01, adapt_steps=100):
    y_pilots, labels_pilots, device_h, device_beta = generate_device_data(P, is_pilot=True)
    adapted_weights = {k: v.clone() for k, v in weights.items()}
    
    if P > 0:
        for _ in range(adapt_steps):
            logits = functional_forward(y_pilots, adapted_weights)
            grads = torch.autograd.grad(F.cross_entropy(logits, labels_pilots), adapted_weights.values())
            adapted_weights = {name: w - lr * g for ((name, w), g) in zip(adapted_weights.items(), grads)}
        
    y_payload, labels_payload, _, _ = generate_device_data(payload_size, h=device_h, beta=device_beta)
    with torch.no_grad():
        preds = torch.argmax(functional_forward(y_payload, adapted_weights), dim=1)
    return torch.sum(preds != labels_payload).item() / payload_size

# ==========================================
# 4. PLOTTING FIGURE 6
# ==========================================
def generate_figure_6(maml_weights, joint_weights, fixed_weights):
    print("\nGenerating Figure 6 (Evaluating devices... this takes a moment)")
    p_values = [0, 2, 4, 8, 12, 14, 16]
    ser_maml, ser_joint, ser_fixed = [], [], []
    num_test_devices = 50 
    
    for P in p_values:
        avg_maml = sum(evaluate_model(maml_weights, P, adapt_steps=100) for _ in range(num_test_devices)) / num_test_devices
        avg_joint = sum(evaluate_model(joint_weights, P, adapt_steps=500) for _ in range(num_test_devices)) / num_test_devices
        avg_fixed = sum(evaluate_model(fixed_weights, P, adapt_steps=500) for _ in range(num_test_devices)) / num_test_devices
            
        ser_maml.append(avg_maml); ser_joint.append(avg_joint); ser_fixed.append(avg_fixed)
        print(f"P={P} | Fixed SER: {avg_fixed:.4f} | Joint SER: {avg_joint:.4f} | MAML SER: {avg_maml:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, ser_fixed, label='fixed initialization', linestyle='--', color='green')
    plt.plot(p_values, ser_joint, label='joint training', linestyle='-', marker='s', markerfacecolor='none', color='blue')
    plt.plot(p_values, ser_maml, label='MAML', linestyle='-', marker='o', markerfacecolor='none', color='red')
    
    plt.yscale('log') 
    plt.xlabel('number of pilots for meta-test device (P)', fontsize=12)
    plt.ylabel('probability of symbol error', fontsize=12)
    plt.title('Figure 6: Demodulation Performance (Realistic Scenario)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/Figure_6.png', dpi=300)
    plt.show()

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
def run_realistic_scenario():
    K = 20
    N_tr = 16 
    N_te = 16
    inner_lr = 0.01
    meta_lr = 0.001
    meta_epochs = 5000  
    
    maml_weights = initialize_meta_weights()
    joint_weights = initialize_meta_weights()
    fixed_weights = initialize_meta_weights()
    
    maml_opt = torch.optim.Adam(maml_weights.values(), lr=meta_lr) 
    joint_opt = torch.optim.Adam(joint_weights.values(), lr=meta_lr)
    
    print("Starting Training (MAML & Joint Baseline)...")
    fixed_devices = [generate_device_data(1)[2:4] for _ in range(K)]
    
    for epoch in range(meta_epochs):
        maml_loss = 0.0; joint_loss = 0.0
        for h_k, beta_k in fixed_devices:
            y_j, labels_j, _, _ = generate_device_data(N_tr + N_te, h=h_k, beta=beta_k, is_pilot=True)
            joint_loss += F.cross_entropy(functional_forward(y_j, joint_weights), labels_j)
            
            y_tr, labels_tr, _, _ = generate_device_data(N_tr, h=h_k, beta=beta_k, is_pilot=True)
            loss_tr = F.cross_entropy(functional_forward(y_tr, maml_weights), labels_tr)
            grads = torch.autograd.grad(loss_tr, maml_weights.values(), create_graph=True)
            fast_w = {name: w - inner_lr * g for ((name, w), g) in zip(maml_weights.items(), grads)}
                
            y_te, labels_te, _, _ = generate_device_data(N_te, h=h_k, beta=beta_k, is_pilot=True)
            maml_loss += F.cross_entropy(functional_forward(y_te, fast_w), labels_te)
            
        joint_opt.zero_grad(); (joint_loss / K).backward(); joint_opt.step()
        maml_opt.zero_grad(); (maml_loss / K).backward(); maml_opt.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1} | MAML Loss: {(maml_loss/K):.4f} | Joint Loss: {(joint_loss/K):.4f}")

    print("Training Complete!")
    generate_figure_6(maml_weights, joint_weights, fixed_weights)

if __name__ == "__main__":
    run_realistic_scenario()