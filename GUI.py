import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import joblib

class PredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Prediction App")
        self.master.geometry("800x600")
        self.master.configure(bg="#1F1F23")  # Updated dark background color

        self.file_label = tk.Label(master, text="Select CSV File:", bg="#1F1F23", fg="#DADADA", font=("Helvetica", 16, "bold"))
        self.file_label.pack(pady=(50, 5), padx=20)

        self.file_button = tk.Button(master, text="Browse", command=self.browse_file, bg="#7289DA", fg="black", font=("Helvetica", 12, "bold"), width=20)
        self.file_button.pack(pady=(0, 10), padx=20)

        self.upload_button = tk.Button(master, text="Upload", command=self.upload_file, bg="#7289DA", fg="black", font=("Helvetica", 12, "bold"), width=20)
        self.upload_button.pack(pady=(0, 10), padx=20)
        self.upload_button.pack_forget()

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_results, bg="#7289DA", fg="black", font=("Helvetica", 12, "bold"), width=20)
        self.clear_button.pack(pady=(0, 10), padx=20)
        self.clear_button.pack_forget()

        self.result_label = tk.Label(master, text="", bg="#1F1F23", fg="#FF0000", font=("Helvetica", 12))
        self.result_label.pack()

        self.prediction_label = tk.Label(master, text="Prediction Results:", bg="#1F1F23", fg="#DADADA", font=("Helvetica", 16, "bold"))
        self.prediction_label.pack(pady=(30, 10))

        self.tree_frame = tk.Frame(master, bg="#1F1F23")
        self.tree_frame.pack(pady=10, padx=20)

        self.tree = ttk.Treeview(self.tree_frame, columns=("Id", "SalePrice"), show="headings", height=15)
        self.tree.heading("Id", text="Id", anchor=tk.CENTER)
        self.tree.heading("SalePrice", text="Sale Price", anchor=tk.CENTER)
        self.tree.pack(side="left", fill="both", expand=True)

        self.tree_scroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=self.tree_scroll.set)

        self.model_path = 'random_forest_regressor_model.joblib'
        self.model = joblib.load(self.model_path)

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.upload_button.pack(pady=(0, 10))

    def upload_file(self):
        if not hasattr(self, 'file_path'):
            self.result_label.config(text="Please select a file first.", fg="#FF0000")
            return

        try:
            data = pd.read_csv(self.file_path)
            X_new = data.drop(['SalePrice'], axis=1, errors='ignore')

            predictions = self.model.predict(X_new.drop(['Id'], axis=1, errors='ignore'))

            results = [{'Id': int(id_val), 'SalePrice': float(pred)} for id_val, pred in zip(X_new['Id'], predictions)]

            self.clear_results()

            for result in results:
                self.tree.insert("", "end", values=(result['Id'], result['SalePrice']))

            self.clear_button.pack(pady=(0, 10))
            self.result_label.config(text="File uploaded successfully.", fg="#32CD32")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", fg="#FF0000")

    def clear_results(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.result_label.config(text="")
        self.upload_button.pack_forget()
        self.clear_button.pack_forget()


def main():
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
