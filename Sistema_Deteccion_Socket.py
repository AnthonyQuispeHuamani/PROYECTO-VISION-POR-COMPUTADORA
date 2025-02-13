import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import mysql.connector
import socket

class CamaraApp:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Aplicación de Cámara")
        self.ventana.geometry("800x400")

        # Inicializar variables
        self.socket_port = tk.IntVar(value=65432)  # Valor por defecto del puerto

        # Frame principal
        self.frame_principal = tk.Frame(ventana)
        self.frame_principal.pack(fill=tk.BOTH, expand=True)

        # Frame para la cámara
        self.frame_camara = tk.Frame(self.frame_principal, width=600, height=600, bg='black')
        self.frame_camara.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame para botones
        self.frame_botones = tk.Frame(self.frame_principal, width=200, height=600)
        self.frame_botones.pack(side=tk.RIGHT, fill=tk.Y)

        # Etiqueta para mostrar video
        self.label_video = tk.Label(self.frame_camara)
        self.label_video.pack(fill=tk.BOTH, expand=True)

        # Crear botones y entradas
        self.crear_botones_y_entradas()

        # Variables de control
        self.captura = None
        self.esta_grabando = False
        self.corriendo = True
        self.hilo_camara = None
        self.socket_validado = False

        # Cargar modelo YOLO
        self.modelo = YOLO("Esmeril.pt.pt")
        self.modelo = YOLO("TALADRO3.pt")

    def crear_botones_y_entradas(self):
        # Entrada para el puerto del socket
        self.label_socket = tk.Label(self.frame_botones, text="Puerto del socket:")
        self.label_socket.pack(pady=5)

        self.entry_socket = tk.Entry(self.frame_botones, textvariable=self.socket_port, width=20)
        self.entry_socket.pack(pady=5)

        # Botón para validar el socket
        self.btn_validar_socket = tk.Button(self.frame_botones, text="Validar Socket", command=self.validar_socket, width=20, height=2)
        self.btn_validar_socket.pack(pady=10)

        # Botón para capturar foto
        self.btn_capturar = tk.Button(self.frame_botones, text="Capturar Foto", command=self.capturar_foto, width=20, height=3, state=tk.DISABLED)
        self.btn_capturar.pack(pady=10)

        # Botón para salir
        self.btn_salir = tk.Button(self.frame_botones, text="Salir", command=self.salir, width=20, height=3)
        self.btn_salir.pack(pady=10)

    def validar_socket(self):
        try:
            port = self.socket_port.get()
            if 1024 <= port <= 65535:
                messagebox.showinfo("Validación Exitosa", f"El puerto {port} es válido.")
                self.socket_validado = True
                self.iniciar_camara()  # Iniciar la cámara después de la validación
                self.btn_capturar.config(state=tk.NORMAL)  # Habilitar el botón de capturar
            else:
                messagebox.showerror("Error", "El puerto debe estar entre 1024 y 65535.")
        except tk.TclError:
            messagebox.showerror("Error", "Ingrese un número válido para el puerto.")

    def iniciar_camara(self):
        if not self.socket_validado:
            messagebox.showerror("Error", "Debe validar el socket antes de iniciar la cámara.")
            return

        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            messagebox.showerror("Error", "No se puede abrir la cámara")
            return

        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, 220)  # Ancho reducido
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 220)  # Altura reducida
        self.captura.set(cv2.CAP_PROP_FPS, 30)

        # Iniciar hilo para mostrar video
        self.hilo_camara = threading.Thread(target=self.mostrar_video, daemon=True)
        self.hilo_camara.start()

    def mostrar_video(self):
        while self.corriendo:
            ret, frame = self.captura.read()
            if not ret:
                break

            # Realizar predicción con el modelo YOLO
            resultados = self.modelo.predict(frame, imgsz=640, conf=0.85)
            anotaciones = resultados[0].plot()

            # Convertir frame a formato adecuado para Tkinter
            frame_rgb = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Actualizar el frame mostrado
            self.ventana.after(1, self.actualizar_frame, imgtk)

    def actualizar_frame(self, imgtk):
        if self.corriendo:
            self.label_video.imgtk = imgtk
            self.label_video.configure(image=imgtk)

    def capturar_foto(self):
        ret, frame = self.captura.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo capturar la foto.")
            return

        # Procesar el frame con el modelo YOLO
        resultados = self.modelo.predict(frame, imgsz=640, conf=0.50)
        anotaciones = resultados[0].plot()

        # Guardar la imagen procesada
        nombre_archivo = "foto_capturada.jpg"
        cv2.imwrite(nombre_archivo, anotaciones)
        messagebox.showinfo("Éxito", f"Foto guardada como {nombre_archivo}")

        # Convertir la imagen a formato binario para la base de datos
        _, buffer = cv2.imencode('.jpg', anotaciones)
        foto_binaria = buffer.tobytes()

        # Guardar en la base de datos
        conexion = conectar_base_datos()
        if conexion:
            try:
                cursor = conexion.cursor()
                # Consulta para insertar la foto en la base de datos
                query = "INSERT INTO fotos_historial (foto_capture) VALUES (%s)"
                cursor.execute(query, (foto_binaria,))
                conexion.commit()

                # Obtener el ID de la última fila insertada
                id_imagen = cursor.lastrowid
                messagebox.showinfo("Éxito", f"Foto guardada en la base de datos con ID: {id_imagen}")

                # Configuración del servidor
                HOST = 'localhost'  # Dirección del servidor
                PORT = self.socket_port.get()  # Obtener el puerto del socket desde la entrada

                # Crear un socket TCP/IP
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.bind((HOST, PORT))
                    server_socket.listen()  # Espera conexiones
                    print("Esperando conexión de Java...")

                    conn, addr = server_socket.accept()  # Aceptar conexión
                    with conn:
                        print(f"Conexión establecida con: {addr}")
                        # Convertir el ID a cadena y enviar
                        mensaje = str(id_imagen)
                        conn.sendall(mensaje.encode('utf-8'))  # Enviar mensaje
                        print(f"Mensaje enviado a Java: {mensaje}")

                messagebox.showinfo("Éxito", f"Dato enviado, ID de la imagen: {mensaje}")

            except mysql.connector.Error as err:
                messagebox.showerror("Error", f"Error al guardar en la base de datos: {err}")
            finally:
                cursor.close()
                conexion.close()

    def salir(self):
        self.corriendo = False
        if self.captura:
            self.captura.release()
        self.ventana.quit()
        self.ventana.destroy()


def conectar_base_datos():
    try:
        conexion = mysql.connector.connect(
            host="bjhuu74zbybbjwwselst-mysql.services.clever-cloud.com",  # Host de CleverCloud
            database="bjhuu74zbybbjwwselst",  # Nombre de la base de datos
            user="udcje5c434qob2xl",  # Usuario
            password="5xX7Bx8BImPoOM5h1Zen",  # Contraseña
            port=3306  # Puerto por defecto de MySQL
        )
        return conexion
    except mysql.connector.Error as err:
        print(f"Error al conectar a la base de datos: {err}")
        return None


# Crear ventana principal
root = tk.Tk()

# Crear aplicación
app = CamaraApp(root)

# Iniciar bucle
root.mainloop()
