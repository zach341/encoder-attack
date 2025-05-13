# encoder-attack

# Abstract
Self-supervised learning has become a powerful paradigm for pre-training encoders on vast amounts
of unlabeled data, providing general-purpose feature extractors for downstream tasks. These pre-trained encoders
are increasingly offered as commercial services, allowing users to fine-tune them for specific tasks. However, the
security of these pre-trained encoders remains underexplored, particularly in black-box settings where attackers have
limited knowledge of the system. In this paper, we propose a novel black-box adversarial attack framework targeting
pre-trained encoders. Our approach generates adversarial examples that are agnostic to the specific downstream
task, allowing for a broad range of attacks without additional query overhead. By training a substitute encoder
with self-supervised learning and aligning adversarial examples, we demonstrate a highly effective transfer attack
method that can compromise downstream tasks, even in scenarios with adversarial defenses or fine-tuned models.
Experimental results on multiple datasets show that our method achieves high attack success rates under black-box
conditions, highlighting the vulnerability of commercial pre-trained encoders to adversarial attacks. These findings
underscore the need for further research into securing pre-trained encoder systems against such threats.

# Experiment
![image](https://github.com/user-attachments/assets/eac8b8a2-870e-4f6d-aaa5-c3cabbd7a680)
![image](https://github.com/user-attachments/assets/714615e2-8f82-4bc2-a531-55845c2a03d2)


