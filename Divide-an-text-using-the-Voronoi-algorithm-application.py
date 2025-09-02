import random
from typing import List, Tuple, Any
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import hashlib


def partition_voronoi(values: List[float], n_seeds: int, lloyd_iters: int = 0, init_method: str = "random", original_items: List[Any] = None) -> Tuple[List[List[Any]], np.ndarray]:
    """Разбивает список чисел values на n_seeds наборов методом 1D-Voronoi с опциональными итерациями Ллойда.
    Если переданы original_items, то итоговые кластеры будут содержать элементы original_items в том же порядке, что и входные значения values.
    Возвращает: clusters - список длины n_seeds; каждый элемент — список элементов original_items (или чисел)
    seeds - массив позиций сайтов (float), длины n_seeds
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")
    if len(values) == 0:
        return [[] for _ in range(n_seeds)], np.array([])

    arr = np.array(values, dtype=float)

    # Инициализация сайтов
    if init_method == "quantile":
        sorted_vals = np.sort(arr)
        probs = np.linspace(0, 1, n_seeds + 2)[1:-1]
        seeds = np.array([np.quantile(sorted_vals, p) for p in probs], dtype=float)
    else:
        lo, hi = arr.min(), arr.max()
        seeds = np.array([random.uniform(lo, hi) for _ in range(n_seeds)], dtype=float)

    # Итерации Ллойда
    for _ in range(int(lloyd_iters)):
        idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
        new_seeds = seeds.copy()
        moved = False
        for s in range(n_seeds):
            members = arr[idx == s]
            if members.size > 0:
                new_pos = members.mean()
                if new_pos != seeds[s]:
                    moved = True
                new_seeds[s] = new_pos
        seeds = new_seeds
        if not moved:
            break

    # Финальное разбиение
    idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
    clusters: List[List[Any]] = [[] for _ in range(n_seeds)]
    for i, k in enumerate(idx):
        item = original_items[i] if original_items is not None else float(values[i])
        clusters[int(k)].append(item)

    return clusters, seeds


# -----------------Helper: mapping текста->числа-----------------

def map_tokens_to_positions(tokens: List[str], method: str) -> List[float]:
    """Преобразует список строк tokens в список чисел (позиции) в соответствии с методом:
    - "index": позиция в исходном порядке (0..n-1)
    - "length": длина строки (кол-во символов)
    - "alphabetical": ранг в сортированном наборе уникальных токенов (0..m-1)
    - "hash": детерминированный числовой хеш, нормализованный
    Возвращает список float той же длины, что и tokens.
    """
    if method == "index":
        return [float(i) for i in range(len(tokens))]
    if method == "length":
        return [float(len(t)) for t in tokens]
    if method == "alphabetical":
        uniq = sorted(set(tokens))
        rank = {v: i for i, v in enumerate(uniq)}
        return [float(rank[t]) for t in tokens]
    if method == "hash":
        # стабильный хеш через sha256 -> int -> normalize
        vals = []
        for t in tokens:
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            num = int(h[:16], 16)  # возьмем первые 16 hex-символов
            vals.append(float(num))
        # нормализуем, чтобы числа не были слишком большими (масштаб останется)
        arr = np.array(vals, dtype=float)
        # сдвинем и масштабируем в [0,1] * len(tokens) для удобства
        if arr.max() == arr.min():
            return arr.tolist()
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        # умножим на n_tokens чтобы получить разброс
        return (norm * float(len(tokens))).tolist()

    # fallback: try to map to float (если строки содержат числа)
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except Exception:
            out.append(0.0)
    return out


# -----------------GUI-----------------

class VoronoiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("1D Voronoi — конвертер (числа или текст)")
        self.geometry("900x660")
        self.create_widgets()

    def create_widgets(self):
        pad = 8
        # Ввод данных
        lbl_values = ttk.Label(self, text="Введите элементы (через пробел, запятую или новую строку):")
        lbl_values.pack(anchor="w", pady=(pad, 0), padx=pad)
        self.txt_values = tk.Text(self, height=8)
        self.txt_values.pack(fill="x", padx=pad)
        self.txt_values.insert("1.0", "apple banana cherry date eggfruit fig grape")

        frm_params = ttk.Frame(self)
        frm_params.pack(fill="x", padx=pad, pady=(6, 0))

        # Тип данных
        ttk.Label(frm_params, text="Тип входных данных:").grid(row=0, column=0, sticky="w")
        self.data_type_var = tk.StringVar(value="text")
        cmb_type = ttk.Combobox(frm_params, textvariable=self.data_type_var, values=["numbers", "text"], state="readonly", width=10)
        cmb_type.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.data_type_var.trace_add("write", self.on_data_type_change)

        # Если текст — метод отображения
        ttk.Label(frm_params, text="Метод для текста:").grid(row=0, column=2, sticky="w")
        self.map_var = tk.StringVar(value="alphabetical")
        self.cmb_map = ttk.Combobox(frm_params, textvariable=self.map_var, values=["index", "length", "alphabetical", "hash"], state="readonly", width=12)
        self.cmb_map.grid(row=0, column=3, sticky="w", padx=(6, 12))

        # Число сайтов
        ttk.Label(frm_params, text="Количество кластеров (n_seeds):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.spin_n = tk.Spinbox(frm_params, from_=1, to=100, width=6)
        self.spin_n.delete(0, "end")
        self.spin_n.insert(0, "3")
        self.spin_n.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(6, 0))

        # Итерации Ллойда
        ttk.Label(frm_params, text="Итераций Ллойда:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.spin_lloyd = tk.Spinbox(frm_params, from_=0, to=100, width=6)
        self.spin_lloyd.delete(0, "end")
        self.spin_lloyd.insert(0, "10")
        self.spin_lloyd.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(6, 0))

        # Метод инициализации
        ttk.Label(frm_params, text="Метод инициализации:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.init_var = tk.StringVar(value="quantile")
        cmb = ttk.Combobox(frm_params, textvariable=self.init_var, values=["quantile", "random"], state="readonly", width=10)
        cmb.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(6, 0))

        # Опция: единый текст
        frm_text_opt = ttk.Frame(self)
        frm_text_opt.pack(fill="x", padx=pad, pady=(6, 0))
        self.single_text_var = tk.BooleanVar(value=False)
        chk_single = ttk.Checkbutton(frm_text_opt, text="Вывести как единый текст", variable=self.single_text_var)
        chk_single.pack(side="left")
        ttk.Label(frm_text_opt, text="Разделитель:").pack(side="left", padx=(12, 6))
        self.sep_var = tk.StringVar(value="space")
        cmb_sep = ttk.Combobox(frm_text_opt, textvariable=self.sep_var, values=["space", "comma", "newline"], state="readonly", width=8)
        cmb_sep.pack(side="left")

        # Кнопки
        frm_buttons = ttk.Frame(self)
        frm_buttons.pack(fill="x", padx=pad, pady=(12, 0))

        btn_load = ttk.Button(frm_buttons, text="Загрузить из файла...", command=self.load_file)
        btn_load.pack(side="left", padx=(0, 6))

        btn_convert = ttk.Button(frm_buttons, text="Конвертировать и сохранить на рабочий стол", command=self.on_convert)
        btn_convert.pack(side="left", padx=(0, 6))

        btn_preview = ttk.Button(frm_buttons, text="Предпросмотр (не сохранять)", command=self.on_preview)
        btn_preview.pack(side="left", padx=(6, 6))

        btn_copy = ttk.Button(frm_buttons, text="Копировать в буфер", command=self.copy_result_to_clipboard)
        btn_copy.pack(side="left")

        # Результат
        lbl_res = ttk.Label(self, text="Результат:")
        lbl_res.pack(anchor="w", pady=(12, 0), padx=pad)
        self.txt_result = tk.Text(self, height=12)
        self.txt_result.pack(fill="both", expand=True, padx=pad, pady=(0, pad))
        self.txt_result.config(state="disabled")

        # Состояние: скрыть map если не текст
        self.on_data_type_change()

    def on_data_type_change(self, *args):
        if self.data_type_var.get() == "text":
            self.cmb_map.state(["!disabled"]) if hasattr(self, 'cmb_map') else None
        else:
            self.cmb_map.state(["disabled"]) if hasattr(self, 'cmb_map') else None

    def parse_input(self):
        raw = self.txt_values.get("1.0", "end").strip()
        if not raw:
            return [], True
        for ch in [",", "\n", "\t", ";"]:
            raw = raw.replace(ch, " ")
        parts = [p for p in raw.split(" ") if p != ""]
        if self.data_type_var.get() == "numbers":
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    raise ValueError(f"Невозможно преобразовать '{p}' в число. Выберите тип 'text' для строк.")
            return vals, True
        else:
            # текстовые элементы
            return parts, False

    def get_desktop_path(self) -> Path:
        p = Path.home() / "Desktop"
        if p.exists():
            return p
        return Path.home()

    def format_output(self, clusters: List[List[Any]], seeds: np.ndarray, single_text: bool = False, sep_key: str = "space") -> str:
        """Форматирует вывод. Если single_text=True, возвращает объединённую строку (без строки Seeds).
        sep_key in {"space","comma","newline"} определяет разделитель.
        """
        if single_text:
            # flatten clusters preserving кластерный порядок
            flat: List[str] = []
            for c in clusters:
                for x in c:
                    flat.append(self._item_plain(x))
            if sep_key == "space":
                sep = " "
            elif sep_key == "comma":
                sep = ", "
            else:
                sep = "\n"
            return sep.join(flat)

        lines = []
        lines.append(f"Seeds: {np.array2string(seeds, precision=6, separator=', ')}")
        for i, c in enumerate(clusters):
            # форматируем элементы (строки в кавычках)
            items = [self._format_item(x) for x in c]
            lines.append(f"Cluster {i} ({len(c)}): {', '.join(items)}")
        return "\n".join(lines)

    def _format_item(self, x: Any) -> str:
        if isinstance(x, str):
            return f'"{x}"'
        if isinstance(x, float):
            if x.is_integer():
                return str(int(x))
            return str(x)
        return str(x)

    def _item_plain(self, x: Any) -> str:
        """Plain text representation (без кавычек), для объединённого вывода."""
        if isinstance(x, str):
            return x
        if isinstance(x, float):
            if x.is_integer():
                return str(int(x))
            return str(x)
        return str(x)

    def save_to_desktop(self, content: str) -> Path:
        desk = self.get_desktop_path()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"voronoi_1d_{ts}.txt"
        path = desk / fname
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def load_file(self):
        """Открывает диалог выбора текстового файла и подставляет его содержимое в поле ввода."""
        file_path = filedialog.askopenfilename(title="Выберите текстовый файл", initialdir=str(Path.home()), filetypes=[("Text files", "*.txt"), ("All files", "*")])
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
        except UnicodeDecodeError:
            # попытка в кодировке cp1251 (часто для русскоязычных файлов)
            try:
                with open(file_path, "r", encoding="cp1251") as f:
                    data = f.read()
            except Exception as e:
                messagebox.showerror("Ошибка чтения файла", f"Не удалось прочитать файл: {e}")
                return
        except Exception as e:
            messagebox.showerror("Ошибка чтения файла", f"Не удалось прочитать файл: {e}")
            return

        # Вставляем в текстовое поле (заменяем содержимое)
        self.txt_values.delete("1.0", "end")
        self.txt_values.insert("1.0", data)
        # Если файл содержит текст, переводим тип в "text"
        self.data_type_var.set("text")
        self.on_data_type_change()

    def on_convert(self):
        try:
            parsed, is_numeric = self.parse_input()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        try:
            n = int(self.spin_n.get())
            lloyd = int(self.spin_lloyd.get())
            init = self.init_var.get()
        except Exception:
            messagebox.showerror("Ошибка параметров", "Проверьте параметры n_seeds и lloyd_iters")
            return

        single = self.single_text_var.get()
        sep_key = self.sep_var.get()

        if not is_numeric:
            tokens: List[str] = parsed  # type: ignore
            if len(tokens) == 0:
                messagebox.showerror("Пустой ввод", "Введите хотя бы один элемент")
                return
            map_method = self.map_var.get()
            positions = map_tokens_to_positions(tokens, map_method)
            if n > len(tokens):
                if not messagebox.askyesno("Мало элементов", "Количество кластеров больше количества элементов. Продолжить?\n(пустые кластеры будут созданы)"):
                    return
            clusters, seeds = partition_voronoi(positions, n, lloyd_iters=lloyd, init_method=init, original_items=tokens)
        else:
            vals: List[float] = parsed  # type: ignore
            if n > len(vals):
                if not messagebox.askyesno("Мало чисел", "Количество кластеров больше количества чисел. Продолжить?\n(пустые кластеры будут созданы)"):
                    return
            clusters, seeds = partition_voronoi(vals, n, lloyd_iters=lloyd, init_method=init, original_items=None)

        out = self.format_output(clusters, seeds, single_text=single, sep_key=sep_key)
        try:
            path = self.save_to_desktop(out)
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл: {e}")
            return
        self._show_result(out)
        messagebox.showinfo("Сохранено", f"Файл сохранён:\n{path}")

    def on_preview(self):
        try:
            parsed, is_numeric = self.parse_input()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        try:
            n = int(self.spin_n.get())
            lloyd = int(self.spin_lloyd.get())
            init = self.init_var.get()
        except Exception:
            messagebox.showerror("Ошибка параметров", "Проверьте параметры n_seeds и lloyd_iters")
            return

        single = self.single_text_var.get()
        sep_key = self.sep_var.get()

        if not is_numeric:
            tokens: List[str] = parsed  # type: ignore
            map_method = self.map_var.get()
            positions = map_tokens_to_positions(tokens, map_method)
            clusters, seeds = partition_voronoi(positions, n, lloyd_iters=lloyd, init_method=init, original_items=tokens)
        else:
            vals: List[float] = parsed  # type: ignore
            clusters, seeds = partition_voronoi(vals, n, lloyd_iters=lloyd, init_method=init, original_items=None)

        out = self.format_output(clusters, seeds, single_text=single, sep_key=sep_key)
        self._show_result(out)

    def _show_result(self, text: str):
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", "end")
        self.txt_result.insert("1.0", text)
        self.txt_result.config(state="disabled")

    def copy_result_to_clipboard(self):
        self.clipboard_clear()
        txt = self.txt_result.get("1.0", "end").rstrip("\n")
        if not txt:
            messagebox.showinfo("Пусто", "Нет текста для копирования")
            return
        self.clipboard_append(txt)
        messagebox.showinfo("Скопировано", "Текст скопирован в буфер обмена")


if __name__ == "__main__":
    app = VoronoiApp()
    app.mainloop()
