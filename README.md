<h1>🧠 Self-Pruning Neural Network</h1>

<h2>📌 Overview</h2>
<p>
This project implements a <b>self-pruning feed-forward neural network</b> trained on the <b>CIFAR-10 dataset</b>.
Unlike traditional pruning (done after training), this model <b>learns to prune itself during training</b> using learnable gates.
</p>

<p>Each weight is associated with a gate value:</p>
<ul>
  <li><b>Gate ≈ 1</b> → Important weight</li>
  <li><b>Gate ≈ 0</b> → Pruned weight</li>
</ul>

<hr/>

<h2>🚀 Key Features</h2>
<ul>
  <li>Custom <b>PrunableLinear layer</b></li>
  <li>Learnable gating mechanism using sigmoid</li>
  <li>Dynamic pruning during training</li>
  <li>Sparsity-aware loss function</li>
  <li>Analysis of <b>accuracy vs sparsity trade-off</b></li>
  <li>Gate distribution visualization</li>
</ul>

<hr/>

<h2>🏗️ Model Design</h2>

<h3>Prunable Linear Layer</h3>
<p>Each layer contains:</p>
<ul>
  <li>Weight matrix</li>
  <li>Bias</li>
  <li>Learnable <code>gate_scores</code></li>
</ul>

<p><b>Forward Pass:</b></p>

<pre><code class="language-python">
gates = torch.sigmoid(gate_scores)
pruned_weights = weight * gates
output = linear(x, pruned_weights, bias)
</code></pre>

<p>
This allows the network to suppress less important connections during training.
</p>

<hr/>

<h2>📉 Loss Function</h2>

<p>The total loss combines classification performance with sparsity regularization:</p>

<pre><code class="language-python">
Total Loss = CrossEntropyLoss + λ * SparsityLoss
</code></pre>

<ul>
  <li><b>CrossEntropyLoss</b> → Measures prediction accuracy</li>
  <li><b>SparsityLoss</b> → Encourages gates to move toward zero</li>
</ul>

<p>Increasing λ leads to stronger pruning.</p>

<hr/>

<h2>📊 Results</h2>

<table>
  <thead>
    <tr>
      <th>Lambda (λ)</th>
      <th>Test Accuracy (%)</th>
      <th>Sparsity (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.0</td>
      <td>53.63</td>
      <td>44.89</td>
    </tr>
    <tr>
      <td>5.0</td>
      <td>54.16</td>
      <td>52.22</td>
    </tr>
    <tr>
      <td>10.0</td>
      <td>54.03</td>
      <td>60.81</td>
    </tr>
  </tbody>
</table>

<hr/>

<h2>📈 Observations</h2>
<ul>
  <li>Sparsity <b>increases</b> as λ increases</li>
  <li>Accuracy remains <b>relatively stable</b></li>
  <li>Best performance observed at <b>λ = 5.0</b></li>
  <li>Higher λ leads to more aggressive pruning with minimal accuracy drop</li>
</ul>

<p>👉 This shows a clear <b>trade-off between model compression and accuracy</b></p>

<hr/>

<h2>📊 Gate Distribution</h2>
<ul>
  <li>A large spike near <b>0</b> (pruned weights)</li>
  <li>A smaller cluster away from 0 (important weights)</li>
</ul>

<hr/>

<h2>⚙️ How to Run</h2>

<pre><code class="language-bash">
pip install torch torchvision matplotlib
python train.py
</code></pre>

<hr/>

<h2>🧪 Evaluation Metrics</h2>
<ul>
  <li><b>Accuracy</b> → Performance on test dataset</li>
  <li><b>Sparsity</b> → Percentage of weights pruned</li>
</ul>

<pre><code class="language-python">
sparsity = (gates &lt; threshold).mean()
</code></pre>

<hr/>

<h2>💡 Key Insight</h2>
<p>
By introducing learnable gates and sparsity regularization, the network can automatically identify and remove unnecessary connections, resulting in a more efficient architecture.
</p>

<hr/>

<h2>🔮 Future Improvements</h2>
<ul>
  <li>Extend to <b>CNN architectures</b></li>
  <li>Apply <b>hard pruning</b> for faster inference</li>
  <li>Deploy using <b>FastAPI</b></li>
  <li>Explore alternative sparsity techniques</li>
</ul>

<hr/>

<h2>📌 Conclusion</h2>
<p>
This project demonstrates that neural networks can dynamically optimize their own structure during training.
Significant sparsity can be achieved with minimal impact on accuracy.
</p>

<hr/>

<p>⭐ If you found this useful, consider starring the repository!</p>
