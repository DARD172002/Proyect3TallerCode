import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import freqz

class IIRFilter:
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self.x_history = [0.0] * len(b)
        self.y_history = [0.0] * len(a)
    
    def process(self, x):
        # Actualizar historial de entradas
        self.x_history.pop()
        self.x_history.insert(0, x)
        
        # Calcular la salida usando la ecuación de diferencias
        y = 0.0
        for i in range(len(self.b)):
            y += self.b[i] * self.x_history[i]
        for i in range(1, len(self.a)):
            y -= self.a[i] * self.y_history[i-1]
        
        # Actualizar historial de salidas
        self.y_history.pop()
        self.y_history.insert(0, y)
        
        return y

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.stream = None
        
        # Coeficientes de los filtros
        # Pasa bajas 4kHz
        self.b_lp = [0.057200, 0.114401, 0.057200]
        self.a_lp = [1.0000, -1.2189, 0.4477]
        
        # Pasa altas 8kHz
        self.b_hp = [0.4316, -0.8632, 0.4316]
        self.a_hp = [1.0000, -0.5087, 0.2177]
        
        # Pasa banda 5-12kHz
        self.b_bp = [0.1523, 0.0, -0.1523]
        self.a_bp = [1.0, -0.7382, 0.3054]
        
        # Elimina banda 4-8kHz
        self.b_bs = [0.6666, -1.8237, 2.5805, -1.8237, 0.6666]
        self.a_bs = [1.0000, -2.2014, 2.4661, -1.4459, 0.4477]
        
        # Pasa bajas configurable (inicializado a 2kHz)
        self.lp_cutoff = 2000
        self.update_lp_configurable(2000)
        
        # Inicializar filtros
        self.lp_filter = IIRFilter(self.b_lp, self.a_lp)
        self.hp_filter = IIRFilter(self.b_hp, self.a_hp)
        self.bp_filter = IIRFilter(self.b_bp, self.a_bp)
        self.bs_filter = IIRFilter(self.b_bs, self.a_bs)
        self.lp_config_filter = IIRFilter(self.b_lp_config, self.a_lp_config)
        
        # Estados de los filtros
        self.lp_active = False
        self.hp_active = False
        self.bp_active = False
        self.bs_active = False
        self.lp_config_active = False
        
    def update_lp_configurable(self, cutoff):
        """Actualiza coeficientes del filtro pasa bajas configurable"""
        self.lp_cutoff = cutoff
        # Diseño simplificado
        fc_normalized = cutoff / (self.sample_rate / 2)
        theta = 2 * np.pi * fc_normalized
        gamma = 2 - np.cos(theta)
        b0 = (1 - np.cos(theta)) / 2
        b1 = 1 - np.cos(theta)
        b2 = b0
        a0 = gamma - np.sqrt(gamma**2 - 1)
        a1 = 2 * (gamma - 1)
        a2 = 1 - a0 - a1
        
        self.b_lp_config = [b0, b1, b2]
        self.a_lp_config = [1.0, -a1, -a2]
        
        if hasattr(self, 'lp_config_filter'):
            self.lp_config_filter = IIRFilter(self.b_lp_config, self.a_lp_config)
    
    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        
        for i in range(frames):
            x = indata[i, 0]
            y = 0.0
            total_filters = 0
            
            if self.lp_active:
                y += self.lp_filter.process(x)
                total_filters += 1
            if self.hp_active:
                y += self.hp_filter.process(x)
                total_filters += 1
            if self.bp_active:
                y += self.bp_filter.process(x)
                total_filters += 1
            if self.bs_active:
                y += self.bs_filter.process(x)
                total_filters += 1
            if self.lp_config_active:
                y += self.lp_config_filter.process(x)
                total_filters += 1
            
            # Promedio si hay múltiples filtros activos
            if total_filters > 0:
                outdata[i, 0] = y / total_filters
            else:
                outdata[i, 0] = x
    
    def start_stream(self):
        try:
            # Intenta con la nueva API primero
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                channels=1,
                callback=self.audio_callback
            )
            self.stream.start()
        except TypeError:
            # Si falla, usa la API antigua
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                channels=1,
                callback=self.audio_callback,
                input=True,
                output=True
            )
    
    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

class EqualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ecualizador Digital")
        self.root.geometry("800x600")
        self.root.configure(bg='#333333')
        
        self.processor = AudioProcessor()
        
        self.create_gui()
        self.plot_frequency_responses()
        
    def create_gui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#333333')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controles
        control_frame = tk.Frame(main_frame, bg='#444444', padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Título
        title_label = tk.Label(
            control_frame, 
            text="Ecualizador Digital", 
            font=('Helvetica', 16, 'bold'),
            fg='white',
            bg='#444444'
        )
        title_label.pack(pady=(0, 10))
        
        # Controles de filtros
        self.create_filter_controls(control_frame)
        
        # Frame del gráfico
        graph_frame = tk.Frame(main_frame, bg='#444444')
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Figura para la respuesta en frecuencia
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor='#444444')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#444444')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white') 
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Botones de inicio/parada
        button_frame = tk.Frame(main_frame, bg='#333333')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        start_button = tk.Button(
            button_frame,
            text="Iniciar Procesamiento",
            command=self.start_processing,
            bg='#4CAF50',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            padx=20,
            pady=10
        )
        start_button.pack(side=tk.LEFT, padx=5)
        
        stop_button = tk.Button(
            button_frame,
            text="Detener Procesamiento",
            command=self.stop_processing,
            bg='#F44336',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            padx=20,
            pady=10
        )
        stop_button.pack(side=tk.LEFT, padx=5)
    
    def create_filter_controls(self, parent):
        # Estilo para los checkbuttons
        style = ttk.Style()
        style.configure('Custom.TCheckbutton', background='#444444', foreground='white', font=('Helvetica', 10))
        
        # Frame para los filtros fijos
        fixed_filters_frame = tk.Frame(parent, bg='#444444')
        fixed_filters_frame.pack(fill=tk.X, pady=5)
        
        # Pasa bajas 4kHz
        self.lp_var = tk.BooleanVar()
        lp_check = ttk.Checkbutton(
            fixed_filters_frame,
            text="Pasa Bajas (4kHz)",
            variable=self.lp_var,
            style='Custom.TCheckbutton',
            command=lambda: self.toggle_filter('lp', self.lp_var.get())
        )
        lp_check.pack(side=tk.LEFT, padx=10)
        
        # Pasa altas 8kHz
        self.hp_var = tk.BooleanVar()
        hp_check = ttk.Checkbutton(
            fixed_filters_frame,
            text="Pasa Altas (8kHz)",
            variable=self.hp_var,
            style='Custom.TCheckbutton',
            command=lambda: self.toggle_filter('hp', self.hp_var.get())
        )
        hp_check.pack(side=tk.LEFT, padx=10)
        
        # Pasa banda 5-12kHz
        self.bp_var = tk.BooleanVar()
        bp_check = ttk.Checkbutton(
            fixed_filters_frame,
            text="Pasa Banda (5-12kHz)",
            variable=self.bp_var,
            style='Custom.TCheckbutton',
            command=lambda: self.toggle_filter('bp', self.bp_var.get())
        )
        bp_check.pack(side=tk.LEFT, padx=10)
        
        # Elimina banda 4-8kHz
        self.bs_var = tk.BooleanVar()
        bs_check = ttk.Checkbutton(
            fixed_filters_frame,
            text="Elimina Banda (4-8kHz)",
            variable=self.bs_var,
            style='Custom.TCheckbutton',
            command=lambda: self.toggle_filter('bs', self.bs_var.get())
        )
        bs_check.pack(side=tk.LEFT, padx=10)
        
        # Frame para el filtro configurable
        config_frame = tk.Frame(parent, bg='#444444')
        config_frame.pack(fill=tk.X, pady=5)
        
        # Pasa bajas configurable
        self.lp_config_var = tk.BooleanVar()
        lp_config_check = ttk.Checkbutton(
            config_frame,
            text="Pasa Bajas Configurable",
            variable=self.lp_config_var,
            style='Custom.TCheckbutton',
            command=lambda: self.toggle_filter('lp_config', self.lp_config_var.get())
        )
        lp_config_check.pack(side=tk.LEFT, padx=10)
        
        # Control deslizante para la frecuencia de corte
        self.cutoff_var = tk.IntVar(value=2000)
        cutoff_label = tk.Label(
            config_frame,
            text="Frecuencia de corte:",
            fg='white',
            bg='#444444',
            font=('Helvetica', 10)
        )
        cutoff_label.pack(side=tk.LEFT, padx=(20, 5))
        
        cutoff_slider = tk.Scale(
            config_frame,
            from_=100,
            to=8000,
            orient=tk.HORIZONTAL,
            variable=self.cutoff_var,
            command=self.update_configurable_filter,
            bg='#444444',
            fg='white',
            highlightthickness=0,
            troughcolor='#666666',
            activebackground='#4CAF50',
            length=200
        )
        cutoff_slider.pack(side=tk.LEFT)
        
        cutoff_value_label = tk.Label(
            config_frame,
            textvariable=self.cutoff_var,
            fg='white',
            bg='#444444',
            font=('Helvetica', 10)
        )
        cutoff_value_label.pack(side=tk.LEFT, padx=(5, 10))
        
        hz_label = tk.Label(
            config_frame,
            text="Hz",
            fg='white',
            bg='#444444',
            font=('Helvetica', 10)
        )
        hz_label.pack(side=tk.LEFT)
    
    def toggle_filter(self, filter_type, active):
        if filter_type == 'lp':
            self.processor.lp_active = active
        elif filter_type == 'hp':
            self.processor.hp_active = active
        elif filter_type == 'bp':
            self.processor.bp_active = active
        elif filter_type == 'bs':
            self.processor.bs_active = active
        elif filter_type == 'lp_config':
            self.processor.lp_config_active = active
        
        self.plot_frequency_responses()
    
    def update_configurable_filter(self, value):
        cutoff = int(value)
        self.processor.update_lp_configurable(cutoff)
        self.plot_frequency_responses()
    
    def plot_frequency_responses(self):
        self.ax.clear()
        
        # Calcular las respuestas en frecuencia
        w = np.logspace(1, np.log10(self.processor.sample_rate/2), 500)
        
        # Pasa bajas 4kHz
        if self.lp_var.get():
            w_lp, h_lp = freqz(self.processor.b_lp, self.processor.a_lp, worN=w, fs=self.processor.sample_rate)
            self.ax.semilogx(w_lp, 20 * np.log10(np.abs(h_lp)), label='Pasa Bajas 4kHz', color='blue')
        
        # Pasa altas 8kHz
        if self.hp_var.get():
            w_hp, h_hp = freqz(self.processor.b_hp, self.processor.a_hp, worN=w, fs=self.processor.sample_rate)
            self.ax.semilogx(w_hp, 20 * np.log10(np.abs(h_hp)), label='Pasa Altas 8kHz', color='red')
        
        # Pasa banda 5-12kHz
        if self.bp_var.get():
            w_bp, h_bp = freqz(self.processor.b_bp, self.processor.a_bp, worN=w, fs=self.processor.sample_rate)
            self.ax.semilogx(w_bp, 20 * np.log10(np.abs(h_bp)), label='Pasa Banda 5-12kHz', color='green')
        
        # Elimina banda 4-8kHz
        if self.bs_var.get():
            w_bs, h_bs = freqz(self.processor.b_bs, self.processor.a_bs, worN=w, fs=self.processor.sample_rate)
            self.ax.semilogx(w_bs, 20 * np.log10(np.abs(h_bs)), label='Elimina Banda 4-8kHz', color='purple')
        
        # Pasa bajas configurable
        if self.lp_config_var.get():
            w_lp_cfg, h_lp_cfg = freqz(self.processor.b_lp_config, self.processor.a_lp_config, worN=w, fs=self.processor.sample_rate)
            self.ax.semilogx(w_lp_cfg, 20 * np.log10(np.abs(h_lp_cfg)), 
                            label=f'Pasa Bajas {self.processor.lp_cutoff}Hz', color='orange')
        
        # Configuración del gráfico
        self.ax.set_title('Respuesta en Frecuencia de los Filtros', color='white')
        self.ax.set_xlabel('Frecuencia (Hz)')
        self.ax.set_ylabel('Ganancia (dB)')
        self.ax.grid(True, which='both', linestyle='--', alpha=0.5)
        self.ax.legend(loc='upper right', facecolor='#444444', labelcolor='white')
        self.ax.set_xlim(20, self.processor.sample_rate/2)
        self.ax.set_ylim(-60, 10)
        
        self.canvas.draw()
    
    def start_processing(self):
        try:
            self.processor.start_stream()
        except Exception as e:
            print(f"Error al iniciar el procesamiento: {e}")
            # Mostrar mensaje de error en la interfaz
            error_label = tk.Label(
                self.root,
                text=f"Error: {str(e)}",
                fg='red',
                bg='#333333',
                font=('Helvetica', 10)
            )
            error_label.pack(pady=10)
            self.root.after(3000, error_label.destroy)  # Eliminar después de 3 segundos
    
    def stop_processing(self):
        self.processor.stop_stream()
    
    def on_closing(self):
        self.stop_processing()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EqualizerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()