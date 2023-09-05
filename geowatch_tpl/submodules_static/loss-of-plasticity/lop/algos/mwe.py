
import torch

class MyMWE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = torch.nn.Linear(2, 3)
        self.stem2 = torch.nn.Linear(2, 3)
        self.body = torch.nn.Linear(3, 3)
        self.head1 = torch.nn.Linear(3, 5)
        self.head2 = torch.nn.Linear(3, 7)

    def forward(self, batch):
        batch_outputs = []
        for item in batch:
            if 'domain1' in item:
                feat = self.stem1(item['domain1'])
            elif 'domain2' in item:
                feat = self.stem2(item['domain2'])
            else:
                raise ValueError

            hidden = self.body(feat)
            logits1 = self.head1(hidden)
            logits2 = self.head2(hidden)
            output = {
                'logits1': logits1,
                'logits2': logits2,
            }
            batch_outputs.append(output)
        return output

def main():
    from torchview import draw_graph
    model = MyMWE()

    batch = [
        {'domain1': torch.Tensor([1, 2])},
        {'domain2': torch.Tensor([1, 2])},
        {'domain1': torch.Tensor([1, 2])},
    ]

    # Verify a normal forward pass works
    output = model(batch)

    # Check if draw_graph works
    model_graph = draw_graph(
        model,
        input_data=batch,
        expand_nested=True,
        hide_inner_tensors=True,
        device='meta', depth=np.inf)
    model_graph.visual_graph.view()
    model_graph.visual_graph.render(format='png')
