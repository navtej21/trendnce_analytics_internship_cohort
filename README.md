<h1>🧠 Self-Pruning Neural Network</h1>

<h2>📌 Overview</h2>
<p>
This project implements a <b>self-pruning feed-forward neural network</b> trained on the <b>CIFAR-10 dataset</b>, where the model learns to 
<b>dynamically remove unnecessary connections during training</b>.
</p>

<p>
Instead of pruning after training, each weight is controlled by a <b>learnable gate</b>:
</p>

<ul>
  <li><b>Gate ≈ 1</b> → Important weight (kept)</li>
  <li><b>Gate ≈ 0</b> → Unimportant weight (pruned)</li>
</ul>

<p>
This allows the network to <b>adapt its own architecture</b> while learning.
</p>

<hr/>

<h2>🚀 Key Features</h2>
<ul>
  <li>Custom <b>PrunableLinear layer</b></li>
  <li>Learnable <b>gating mechanism (sigmoid-based)</b></li>
  <li>Dynamic pruning during training</li>
  <li>Custom <b>sparsity-inducing loss</b></li>
  <li>Trade-off analysis between <b>accuracy and sparsity</b></li>
  <li>Visualization of gate distributions</li>
</ul>

<hr/>

<h2>🏗️ Model Architecture</h2>

<h3>Prunable Linear Layer</h3>
<p>Each layer consists of:</p>
<ul>
  <li>Weight matrix</li>
  <li>Bias</li>
  <li>Learnable <code>gate_scores</code> (same shape as weights)</li>
</ul>

<h3>Forward Pass</h3>
<pre><code class="language-python">
gates = torch.sigmoid(gate_scores)
pruned_weights = weight * gates
output = F.linear(x, pruned_weights, bias)
</code></pre>

<p>👉 This ensures only important connections contribute to predictions.</p>

<hr/>

<h2>📉 Loss Function</h2>

<p>The training objective combines classification loss with a sparsity penalty:</p>

<pre><code class="language-python">
Total Loss = CrossEntropyLoss + λ * SparsityLoss
</code></pre>

<ul>
  <li><b>CrossEntropyLoss</b> → Measures prediction accuracy</li>
  <li><b>SparsityLoss</b> → Encourages smaller gate values (promotes pruning)</li>
</ul>

<h3>Sparsity Loss Used</h3>
<pre><code class="language-python">
loss += torch.mean(torch.clamp(gates - 0.05, min=0) ** 2)
</code></pre>

<p>👉 This is a <b>custom penalty (not pure L1)</b> that pushes gates toward zero.</p>

<hr/>

<h2>📊 Experimental Results</h2>

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
  <li>Sparsity <b>increases as λ increases</b></li>
  <li>Accuracy remains <b>stable (~53–54%)</b></li>
  <li>Best accuracy achieved at <b>λ = 5.0</b></li>
  <li>Higher λ results in more aggressive pruning</li>
</ul>

<p>👉 This clearly demonstrates the <b>trade-off between model compression and performance</b></p>

<hr/>

<h2>📊 Sparsity Calculation</h2>

<pre><code class="language-python">
sparsity = (gates &lt; 0.1).mean()
</code></pre>

<ul>
  <li>Threshold = <b>0.1</b></li>
  <li>Represents percentage of effectively pruned weights</li>
</ul>

<hr/>

<h2>📊 Gate Distribution</h2>
<ul>
  <li>A large spike near <b>0</b> → pruned connections</li>
  <li>A smaller cluster away from 0 → important connections</li>
</ul>

<p>This confirms that the model successfully learns a <b>sparse structure</b>.</p>

<hr/>

<h2>⚙️ How to Run</h2>

<pre><code class="language-bash">
pip install torch torchvision matplotlib
python train.py
</code></pre>

<hr/>

<h2>🧪 Evaluation Metrics</h2>
<ul>
  <li><b>Accuracy</b> → Classification performance on CIFAR-10</li>
  <li><b>Sparsity (%)</b> → Percentage of weights pruned</li>
</ul>

<hr/>

<h2>💡 Key Insight</h2>
<ul>
  <li>Automatically identifies redundant connections</li>
  <li>Reduces model complexity</li>
  <li>Maintains competitive accuracy</li>
</ul>

<hr/>

<h2>🔮 Future Improvements</h2>
<ul>
  <li>Extend to <b>Convolutional Neural Networks (CNNs)</b></li>
  <li>Apply <b>hard pruning</b> for inference speedup</li>
  <li>Replace custom loss with <b>true L1 regularization</b></li>
  <li>Tune sparsity threshold for more precise pruning</li>
  <li>Deploy using <b>FastAPI</b></li>
</ul>

<hr/>

<h2>📌 Conclusion</h2>
<p>
This project demonstrates that neural networks can <b>self-optimize their structure during training</b>.
Significant sparsity (~60%) is achieved with <b>minimal loss in accuracy</b>, making the model more efficient without sacrificing performance.
</p>

<hr/>

