import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Configuración inicial
fs = 44100  # Frecuencia de muestreo
block_size = 1024  # Tamaño del bloque de audio
stream = None  # Stream de audio
audio_file = None  # Archivo de audio cargado
file_position = 0  # Posición en el archivo
is_playing_file = False  # Estado de reproducción
last_audio_chunk = np.zeros(block_size)  # Último chunk de audio (para el espectro)
volume = 1.0  # Volumen (0.0 a 1.0)
latency = 0.1  # Latencia en segundos

# Coeficientes para los filtros (inicialmente vacíos)
# Los coeficientes se calcularán cuando se activen los filtros
coeffs_low = {'b': [], 'a': [], 'x_hist': np.zeros(4), 'y_hist': np.zeros(4)}
coeffs_high = {'b': [], 'a': [], 'x_hist': np.zeros(4), 'y_hist': np.zeros(4)}
coeffs_band = {'b': [], 'a': [], 'x_hist': np.zeros(4), 'y_hist': np.zeros(4)}
coeffs_stop = {'b': [], 'a': [], 'x_hist': np.zeros(4), 'y_hist': np.zeros(4)}
coeffs_custom = {'b': [], 'a': [], 'x_hist': np.zeros(2), 'y_hist': np.zeros(2)}
fc_custom = 4000  # Frecuencia de corte personalizada (inicial: 4kHz)

def design_lowpass(fc, fs=44100):
    """Diseña un filtro pasa bajas de 4to orden usando la transformación bilineal."""
    # Pre-warping de la frecuencia
    omega_c = 2 * fs * np.tan(np.pi * fc / fs)
    
    # Diseño del filtro analógico (Butterworth)
    k = omega_c / (omega_c + 2 * fs)
    b0 = k**2
    b1 = 2 * b0
    b2 = b0
    a0 = 1
    a1 = 2 * (k**2 - 1) / (1 + 2*k + k**2)
    a2 = (1 - 2*k + k**2) / (1 + 2*k + k**2)
    
    # Coeficientes para un filtro de 4to orden (dos secciones de 2do orden)
    b = [b0, b1, b2]
    a = [a0, a1, a2]
    
    # Aplicamos dos veces para obtener 4to orden
    return {'b': np.convolve(b, b), 'a': np.convolve(a, a)}

def design_highpass(fc, fs=44100):
    """Diseña un filtro pasa altas de 4to orden usando la transformación bilineal."""
    # Pre-warping de la frecuencia
    omega_c = 2 * fs * np.tan(np.pi * fc / fs)
    
    # Diseño del filtro analógico (Butterworth)
    k = 2 * fs / (omega_c + 2 * fs)
    b0 = k**2
    b1 = -2 * b0
    b2 = b0
    a0 = 1
    a1 = 2 * (k**2 - 1) / (1 + 2*k + k**2)
    a2 = (1 - 2*k + k**2) / (1 + 2*k + k**2)
    
    # Coeficientes para un filtro de 4to orden (dos secciones de 2do orden)
    b = [b0, b1, b2]
    a = [a0, a1, a2]
    
    # Aplicamos dos veces para obtener 4to orden
    return {'b': np.convolve(b, b), 'a': np.convolve(a, a)}

def design_bandpass(f_low, f_high, fs=44100):
    """Diseña un filtro pasa banda de 4to orden."""
    # Diseñamos un pasa bajas y un pasa altas y los combinamos
    low = design_lowpass(f_high, fs)
    high = design_highpass(f_low, fs)
    
    # Combinamos los coeficientes (convolución de las funciones de transferencia)
    b = np.convolve(low['b'], high['b'])
    a = np.convolve(low['a'], high['a'])
    
    return {'b': b, 'a': a}

def design_bandstop(f_low, f_high, fs=44100):
    """Diseña un filtro rechaza banda de 4to orden."""
    # Diseñamos un pasa bajas y un pasa altas y los sumamos
    low = design_lowpass(f_low, fs)
    high = design_highpass(f_high, fs)
    
    # Sumamos las funciones de transferencia
    # Para sumar, los denominadores deben ser iguales (a1*a2)
    # y los numeradores (b1*a2 + b2*a1)
    a = np.convolve(low['a'], high['a'])
    b = np.convolve(low['b'], high['a']) + np.convolve(high['b'], low['a'])
    
    return {'b': b, 'a': a}

def design_custom_lowpass(fc, fs=44100, order=2):
    """Diseña un filtro pasa bajas de 2do orden."""
    # Pre-warping de la frecuencia
    omega_c = 2 * fs * np.tan(np.pi * fc / fs)
    
    # Diseño del filtro analógico (Butterworth)
    k = omega_c / (omega_c + 2 * fs)
    b0 = k**2
    b1 = 2 * b0
    b2 = b0
    a0 = 1
    a1 = 2 * (k**2 - 1) / (1 + 2*k + k**2)
    a2 = (1 - 2*k + k**2) / (1 + 2*k + k**2)
    
    return {'b': np.array([b0, b1, b2]), 'a': np.array([a0, a1, a2])}

def difference_equation_filter(x, coeffs):
    """Aplica un filtro IIR usando la ecuación de diferencias."""
    b = coeffs['b']
    a = coeffs['a']
    x_hist = coeffs['x_hist']
    y_hist = coeffs['y_hist']
    
    y = np.zeros_like(x)
    for n in range(len(x)):
        # Actualizamos el historial de x (entradas)
        x_hist = np.roll(x_hist, 1)
        x_hist[0] = x[n]
        
        # Calculamos la salida
        y[n] = np.sum(b * x_hist[:len(b)]) - np.sum(a[1:] * y_hist[:len(a)-1])
        y[n] /= a[0]
        
        # Actualizamos el historial de y (salidas)
        y_hist = np.roll(y_hist, 1)
        y_hist[0] = y[n]
    
    # Guardamos el historial para la próxima llamada
    coeffs['x_hist'] = x_hist
    coeffs['y_hist'] = y_hist
    
    return y

def apply_filters(audio_data):
    """Aplica todos los filtros activos en cascada."""
    global fc_custom, coeffs_custom
    
    try:
        # Convertimos a float32 para mejor rendimiento
        audio = audio_data.astype(np.float32).copy()

        # Aplicamos los filtros solo si hay señal
        if not np.any(audio):
            return audio

        # Aplicamos los filtros en orden
        if var_lowpass.get():
            if len(coeffs_low['b']) == 0:  # Diseñamos el filtro solo si no está diseñado
                coeffs = design_lowpass(4000, fs)
                coeffs_low['b'] = coeffs['b']
                coeffs_low['a'] = coeffs['a']
            audio = difference_equation_filter(audio, coeffs_low)
            
        if var_highpass.get():
            if len(coeffs_high['b']) == 0:
                coeffs = design_highpass(8000, fs)
                coeffs_high['b'] = coeffs['b']
                coeffs_high['a'] = coeffs['a']
            audio = difference_equation_filter(audio, coeffs_high)
            
        if var_bandpass.get():
            if len(coeffs_band['b']) == 0:
                coeffs = design_bandpass(5000, 12000, fs)
                coeffs_band['b'] = coeffs['b']
                coeffs_band['a'] = coeffs['a']
            audio = difference_equation_filter(audio, coeffs_band)
            
        if var_bandstop.get():
            if len(coeffs_stop['b']) == 0:
                coeffs = design_bandstop(4000, 8000, fs)
                coeffs_stop['b'] = coeffs['b']
                coeffs_stop['a'] = coeffs['a']
            audio = difference_equation_filter(audio, coeffs_stop)
            
        if var_custom.get():
            new_fc = slider_fc.get()
            if new_fc != fc_custom or len(coeffs_custom['b']) == 0:
                fc_custom = new_fc
                coeffs = design_custom_lowpass(fc_custom, fs, 2)
                coeffs_custom['b'] = coeffs['b']
                coeffs_custom['a'] = coeffs['a']
            audio = difference_equation_filter(audio, coeffs_custom)

        # Aplicamos el volumen y prevenimos clipping
        return np.clip(audio * volume, -1.0, 1.0)
    
    except Exception as e:
        print(f"Error en apply_filters: {e}")
        return audio_data

def audio_callback(indata, outdata, frames, time, status):
    """Callback para procesamiento de audio en tiempo real."""
    global file_position, audio_file, is_playing_file, last_audio_chunk

    if status:
        # Solo imprimimos errores críticos
        if status.input_overflow or status.output_underflow:
            return
        print(f"Error crítico en el stream: {status}")

    try:
        if is_playing_file and audio_file is not None:
            remaining_samples = len(audio_file) - file_position
            if remaining_samples <= 0:
                outdata.fill(0)
                is_playing_file = False
                root.after(0, lambda: btn_play_file.config(text="Reproducir Archivo"))
                return
            
            chunk = audio_file[file_position:file_position + frames]
            file_position += len(chunk)
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
            processed_audio = apply_filters(chunk)
            
        else:
            if np.any(indata):
                processed_audio = apply_filters(indata[:, 0])
            else:
                outdata.fill(0)
                return

        # Prevenir clipping
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        outdata[:] = processed_audio.reshape(-1, 1)
        last_audio_chunk = processed_audio

    except Exception as e:
        outdata.fill(0)
        print(f"Error en el procesamiento de audio: {e}")

def load_audio_file():
    """Carga un archivo de audio y lo normaliza."""
    global audio_file, file_position, is_playing_file
    filepath = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")])
    
    if not filepath:
        return

    try:
        # Leemos el archivo manualmente ya que no usamos scipy.io.wavfile
        with open(filepath, 'rb') as f:
            # Leemos el encabezado del archivo WAV (simplificado)
            riff = f.read(4)
            if riff != b'RIFF':
                raise ValueError("No es un archivo WAV válido")
            
            f.read(4)  # Tamaño del archivo - 8
            wave = f.read(4)
            if wave != b'WAVE':
                raise ValueError("No es un archivo WAV válido")
                
            fmt = f.read(4)
            if fmt != b'fmt ':
                raise ValueError("No se encontró el chunk fmt")
                
            fmt_size = int.from_bytes(f.read(4), byteorder='little')
            audio_format = int.from_bytes(f.read(2), byteorder='little')
            num_channels = int.from_bytes(f.read(2), byteorder='little')
            file_fs = int.from_bytes(f.read(4), byteorder='little')
            f.read(4)  # Byte rate
            f.read(2)  # Block align
            bits_per_sample = int.from_bytes(f.read(2), byteorder='little')
            
            # Buscamos el chunk de datos
            while True:
                chunk_id = f.read(4)
                if not chunk_id:
                    raise ValueError("No se encontró el chunk de datos")
                if chunk_id == b'data':
                    break
                chunk_size = int.from_bytes(f.read(4), byteorder='little')
                f.seek(chunk_size, 1)
                
            data_size = int.from_bytes(f.read(4), byteorder='little')
            
            # Leemos los datos de audio
            raw_data = f.read(data_size)
            
        # Convertimos los datos a numpy array según el formato
        if bits_per_sample == 16:
            data = np.frombuffer(raw_data, dtype=np.int16)
            data = data.astype(np.float32) / 32768.0
        elif bits_per_sample == 32:
            data = np.frombuffer(raw_data, dtype=np.float32)
        else:
            raise ValueError("Formato de bits no soportado")
        
        # Convertimos a mono si es estéreo
        if num_channels > 1:
            data = data.reshape(-1, num_channels)
            data = np.mean(data, axis=1)
        
        # Normalizamos
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data /= max_val
        
        # Resampleamos si es necesario (método simple)
        if file_fs != fs:
            ratio = fs / file_fs
            new_length = int(len(data) * ratio)
            data = np.interp(
                np.linspace(0, len(data), new_length, endpoint=False),
                np.arange(len(data)),
                data
            )
        
        audio_file = data
        file_position = 0
        lbl_file.config(text=f"Archivo: {os.path.basename(filepath)}")
        btn_play_file.config(state=tk.NORMAL)
        messagebox.showinfo("Éxito", "Archivo cargado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

def toggle_file_playback():
    """Inicia/detiene la reproducción del archivo."""
    global is_playing_file
    is_playing_file = not is_playing_file
    btn_play_file.config(text="Detener Archivo" if is_playing_file else "Reproducir Archivo")

def toggle_live_processing():
    """Inicia/detiene el procesamiento en vivo del micrófono."""
    global stream
    if stream is None:
        try:
            stream = sd.Stream(
                callback=audio_callback,
                blocksize=block_size,
                samplerate=fs,
                channels=1,
                latency=latency,
                device=None,
                dtype=np.float32
            )
            stream.start()
            btn_live.config(text="Detener Micrófono")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el micrófono: {e}")
    else:
        try:
            stream.stop()
            stream.close()
            stream = None
            btn_live.config(text="Iniciar Micrófono")
        except Exception as e:
            messagebox.showerror("Error", f"Error al detener el micrófono: {e}")

def export_audio():
    """Exporta el audio procesado a un archivo WAV."""
    if audio_file is None:
        messagebox.showwarning("Advertencia", "No hay ningún archivo cargado.")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("Archivos WAV", "*.wav")]
    )
    
    if not filepath:
        return

    try:
        processed_audio = apply_filters(audio_file)
        
        # Escribimos el archivo WAV manualmente
        with open(filepath, 'wb') as f:
            # Encabezado RIFF
            f.write(b'RIFF')
            
            # Tamaño del archivo (se actualizará después)
            file_size_pos = f.tell()
            f.write((0).to_bytes(4, byteorder='little'))
            
            # Formato WAVE
            f.write(b'WAVE')
            
            # Chunk fmt
            f.write(b'fmt ')
            f.write((16).to_bytes(4, byteorder='little'))  # Tamaño del chunk fmt
            f.write((3).to_bytes(2, byteorder='little'))  # Formato (3 = float)
            f.write((1).to_bytes(2, byteorder='little'))  # Canales
            f.write((fs).to_bytes(4, byteorder='little'))  # Frecuencia de muestreo
            f.write((fs * 4).to_bytes(4, byteorder='little'))  # Byte rate
            f.write((4).to_bytes(2, byteorder='little'))  # Block align
            f.write((32).to_bytes(2, byteorder='little'))  # Bits por muestra
            
            # Chunk data
            f.write(b'data')
            data_size_pos = f.tell()
            f.write((0).to_bytes(4, byteorder='little'))  # Tamaño del chunk (se actualizará)
            
            # Escribimos los datos de audio
            data_bytes = processed_audio.astype(np.float32).tobytes()
            f.write(data_bytes)
            
            # Actualizamos los tamaños en el encabezado
            file_size = f.tell() - 8
            data_size = len(data_bytes)
            
            f.seek(file_size_pos)
            f.write(file_size.to_bytes(4, byteorder='little'))
            
            f.seek(data_size_pos)
            f.write(data_size.to_bytes(4, byteorder='little'))
        
        messagebox.showinfo("Éxito", "Audio exportado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo exportar el audio: {e}")

def update_volume(val):
    """Actualiza el volumen global."""
    global volume
    volume = float(val)
    lbl_volume.config(text=f"Volumen: {int(volume * 100)}%")

def update_spectrum():
    """Actualiza el espectro de frecuencia en tiempo real."""
    if np.any(last_audio_chunk):
        # Implementación manual de la FFT
        N = len(last_audio_chunk)
        fft_data = np.abs(manual_fft(last_audio_chunk))
        fft_data /= np.max(fft_data) if np.max(fft_data) > 0 else 1.0
        freqs = np.linspace(0, fs/2, N//2 + 1)
        line.set_data(freqs, fft_data[:N//2 + 1])
        ax.set_xlim(0, fs/2)
        ax.set_ylim(0, 1)
        canvas.draw()
    root.after(100, update_spectrum)

def manual_fft(x):
    """Implementación básica de la FFT (no optimizada)."""
    N = len(x)
    if N <= 1:
        return x
    even = manual_fft(x[0::2])
    odd = manual_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

# --- Interfaz gráfica ---
root = tk.Tk()
root.title("Ecualizador Digital - ITCR (Ecuaciones de Diferencias)")
root.geometry("800x800")

# Variables de control
var_lowpass = tk.BooleanVar()
var_highpass = tk.BooleanVar()
var_bandpass = tk.BooleanVar()
var_bandstop = tk.BooleanVar()
var_custom = tk.BooleanVar()

# Frame principal
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Sección de archivo
file_frame = ttk.LabelFrame(main_frame, text="Archivo de Audio", padding="10")
file_frame.pack(fill=tk.X, pady=5)
btn_load = ttk.Button(file_frame, text="Cargar Archivo", command=load_audio_file)
btn_load.pack(side=tk.LEFT, padx=5)
btn_play_file = ttk.Button(file_frame, text="Reproducir Archivo", command=toggle_file_playback, state=tk.DISABLED)
btn_play_file.pack(side=tk.LEFT, padx=5)
lbl_file = ttk.Label(file_frame, text="Ningún archivo cargado")
lbl_file.pack(side=tk.LEFT, padx=5)

# Sección de micrófono
live_frame = ttk.LabelFrame(main_frame, text="Procesamiento en Vivo", padding="10")
live_frame.pack(fill=tk.X, pady=5)
btn_live = ttk.Button(live_frame, text="Iniciar Micrófono", command=toggle_live_processing)
btn_live.pack()

# Sección de volumen
volume_frame = ttk.LabelFrame(main_frame, text="Volumen", padding="10")
volume_frame.pack(fill=tk.X, pady=5)

lbl_volume = ttk.Label(volume_frame, text="Volumen: 100%")
lbl_volume.pack()

slider_volume = ttk.Scale(volume_frame, from_=0.0, to=1.0, orient="horizontal", command=update_volume)
slider_volume.set(1.0)
slider_volume.pack(fill=tk.X, padx=5, pady=2)

# Sección de filtros
filters_frame = ttk.LabelFrame(main_frame, text="Filtros", padding="10")
filters_frame.pack(fill=tk.BOTH, expand=True, pady=5)

def create_filter_control(parent, text, var):
    """Crea un checkbox para un filtro."""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=2)
    ttk.Checkbutton(frame, text=text, variable=var).pack(side=tk.LEFT)
    return frame

create_filter_control(filters_frame, "Pasa Bajas (<4kHz)", var_lowpass)
create_filter_control(filters_frame, "Pasa Altas (>8kHz)", var_highpass)
create_filter_control(filters_frame, "Pasa Banda (5k-12kHz)", var_bandpass)
create_filter_control(filters_frame, "Rechaza Banda (4k-8kHz)", var_bandstop)

# Filtro personalizado
custom_frame = ttk.Frame(filters_frame)
custom_frame.pack(fill=tk.X, pady=5)
ttk.Checkbutton(custom_frame, text="Pasa Bajas Configurable", variable=var_custom).pack(side=tk.LEFT)
ttk.Label(custom_frame, text="Frecuencia (Hz):").pack(side=tk.LEFT, padx=5)
slider_fc = ttk.Scale(custom_frame, from_=1000, to=8000, orient="horizontal")
slider_fc.set(fc_custom)
slider_fc.pack(side=tk.LEFT, expand=True, fill=tk.X)

# Botón de exportación
btn_export = ttk.Button(main_frame, text="Exportar Audio Procesado", command=export_audio)
btn_export.pack(pady=10)

# Visualizador de espectro
spectrum_frame = ttk.LabelFrame(main_frame, text="Espectro de Frecuencia", padding="10")
spectrum_frame.pack(fill=tk.BOTH, expand=True)

fig = Figure(figsize=(6, 3), dpi=100)
ax = fig.add_subplot(111)
line, = ax.plot([], [])
ax.set_xlim(0, fs / 2)
ax.set_ylim(0, 1)
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Magnitud")
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=spectrum_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Inicia la actualización del espectro
update_spectrum()

root.mainloop()