import torch
import torch.nn as nn
import tkinter as tk
import threading

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


model = NN(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()


class DrawingWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST")
        self.root.geometry("560x640")
        self.root.maxsize(560,640)
        self.root.minsize(560,640)
        self.pixel_size = 20
        self.grid_size = 28

        self.canvas = tk.Canvas(self.root, width=self.pixel_size * self.grid_size, height=self.pixel_size * self.grid_size, bg="white", relief="flat", bd=0, highlightthickness=0)
        self.canvas.place(x=0, y=0)

        self.rs = tk.Button(self.root, text="Reset", font=("Century Gothic", 20), command=self.reset_canvas, relief="flat", bg="lightgrey")
        self.rs.place(relx=0.05, rely=0.89, anchor='nw')

        self.preLabel = tk.Label(self.root, text="You drew: ", font=("Century Gothic", 20))
        self.preLabel.place(relx=0.45, rely=0.91, anchor='nw')

        self.canvas.bind("<B1-Motion>", self.paint)

        self.grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.last_x, self.last_y = None, None

        self.guess()

    def paint(self, event):
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if not self.grid[y][x]:
                self.grid[y][x] = True
                self.canvas.create_rectangle(x * self.pixel_size, y * self.pixel_size,
                                             (x + 1) * self.pixel_size, (y + 1) * self.pixel_size,
                                             fill="black", outline="black")

    def reset_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.pixel_size * self.grid_size, self.pixel_size * self.grid_size, fill="white")
        self.grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]


    def guess(self):
        def sub_process():
            while True:
                if all(all(element == False for element in row) for row in self.grid):
                    self.preLabel.config(text="You drew: nothing")
                else:
                    tensor = torch.tensor(self.grid, dtype=torch.float)
                    tensor = tensor.reshape(-1, 28*28).to(device)
                    out = model(tensor)
                    predicted = torch.max(out,1)
                    self.preLabel.config(text=f"You drew: {predicted[1].item()}")
                if not self.root.winfo_exists():
                    break

        threading.Thread(target=sub_process).start()


root = tk.Tk()
app = DrawingWidget(root)
root.mainloop()