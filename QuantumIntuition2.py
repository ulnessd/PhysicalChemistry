# quantum_visualizer_pyqt6.py

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QMessageBox, QComboBox, QCheckBox,
    QSpinBox
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from scipy.interpolate import UnivariateSpline
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from PyQt6.QtGui import QPainter, QPen, QPainterPath, QPalette, QColor


class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        # --- CHANGE: Use a more direct method to ensure background color is set ---
        self.setAutoFillBackground(True)  # Important for paintEvent background handling
        self.drawing = False
        self.points = []
        self.enabled = True
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event):
        if self.enabled and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.points = [event.position()]
            self.update()

    def mouseMoveEvent(self, event):
        if self.enabled and self.drawing:
            self.points.append(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.enabled and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # --- CHANGE: Explicitly fill the background first ---
        # This is a more robust way to set the color than stylesheets or palettes alone.
        painter.fillRect(self.rect(), QColor('#cccccc'))

        # Draw the border
        border_pen = QPen(QColor("#555"), 1)
        painter.setPen(border_pen)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))  # Adjust to draw inside the widget bounds

        # Draw the user's line
        pen = QPen(Qt.GlobalColor.black, 2)
        painter.setPen(pen)
        if self.points:
            path = QPainterPath()
            path.moveTo(self.points[0])
            for point in self.points[1:]:
                path.lineTo(point)
            painter.drawPath(path)

    def get_physical_points(self):
        if not self.points:
            return None, None
        width = self.width()
        height = self.height()
        x_pixels = np.array([point.x() for point in self.points])
        y_pixels = np.array([point.y() for point in self.points])
        x_min, x_max = -10, 10
        x_physical = x_pixels / width * (x_max - x_min) + x_min
        V_min, V_max = 0, 10
        y_physical = (1 - y_pixels / height) * (V_max - V_min) + V_min
        return x_physical, y_physical

    def clear(self):
        self.points = []
        self.update()


class QuantumVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Potential Visualizer (PyQt6)")
        self.setGeometry(100, 100, 1200, 600)
        self.mass = 1.0
        self.wavefunc_scale = 0.3
        self.plot_probability_density = False
        self.num_states = 5
        self.grid_points = 250
        self.initUI()

    def initUI(self):
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        instructions = QLabel("Draw a potential, select a preset, adjust parameters, then click Compute.")
        left_layout.addWidget(instructions)

        self.canvas = DrawingCanvas()
        left_layout.addWidget(self.canvas)

        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset Potential:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom Drawing", "Square Root |x|", "Absolute Value |x|",
            "Harmonic Oscillator x^2", "Absolute Cubed |x|^3", "Quartic x^4",
            "Lennard-Jones Potential", "Pöschl-Teller Potential"
        ])
        self.preset_combo.currentIndexChanged.connect(self.preset_potential_changed)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        left_layout.addLayout(preset_layout)

        slider_layout = QHBoxLayout()
        mass_label = QLabel("Mass:")
        self.mass_slider = QSlider(Qt.Orientation.Horizontal)
        self.mass_slider.setRange(1, 100)
        self.mass_slider.setValue(10)
        self.mass_slider.valueChanged.connect(self.update_mass)
        slider_layout.addWidget(mass_label)
        slider_layout.addWidget(self.mass_slider)
        left_layout.addLayout(slider_layout)

        wf_slider_layout = QHBoxLayout()
        wf_scale_label = QLabel("Wavefunction Scale:")
        self.wf_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.wf_scale_slider.setRange(1, 100)
        self.wf_scale_slider.setValue(30)
        self.wf_scale_slider.valueChanged.connect(self.update_wf_scale)
        wf_slider_layout.addWidget(wf_scale_label)
        wf_slider_layout.addWidget(self.wf_scale_slider)
        left_layout.addLayout(wf_slider_layout)

        num_states_layout = QHBoxLayout()
        num_states_label = QLabel("Number of States:")
        self.num_states_spinbox = QSpinBox()
        self.num_states_spinbox.setRange(1, 15)
        self.num_states_spinbox.setValue(self.num_states)
        self.num_states_spinbox.valueChanged.connect(self.update_num_states)
        num_states_layout.addWidget(num_states_label)
        num_states_layout.addWidget(self.num_states_spinbox)
        left_layout.addLayout(num_states_layout)

        grid_points_layout = QHBoxLayout()
        grid_points_label = QLabel("Grid Points:")
        self.grid_points_combo = QComboBox()
        self.grid_points_options = [150, 250, 350, 500, 600]
        self.grid_points_combo.addItems([str(p) for p in self.grid_points_options])
        self.grid_points_combo.setCurrentText(str(self.grid_points))
        self.grid_points_combo.currentIndexChanged.connect(self.update_grid_points)
        grid_points_layout.addWidget(grid_points_label)
        grid_points_layout.addWidget(self.grid_points_combo)
        left_layout.addLayout(grid_points_layout)

        self.plot_toggle_checkbox = QCheckBox("Plot Probability Density |ψ|²")
        self.plot_toggle_checkbox.stateChanged.connect(self.update_plot_toggle)
        left_layout.addWidget(self.plot_toggle_checkbox)

        controls_layout = QHBoxLayout()
        compute_button = QPushButton("Compute")
        compute_button.clicked.connect(self.compute_and_plot)
        clear_button = QPushButton("Clear Drawing")
        clear_button.clicked.connect(self.canvas.clear)
        controls_layout.addWidget(compute_button)
        controls_layout.addWidget(clear_button)
        left_layout.addLayout(controls_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Energy')
        self.plot_widget.setLabel('bottom', 'Position')
        self.plot_widget.addLegend()
        right_layout.addWidget(self.plot_widget)

        # Initialize
        self.update_mass(self.mass_slider.value())
        self.preset_potential_changed(0)

    def update_mass(self, value):
        self.mass = value / 10.0
        if self.mass < 0.1: self.mass = 0.1
        self.run_computation(quality='draft')

    def update_wf_scale(self, value):
        self.wavefunc_scale = value / 100.0
        self.plot_results()

    def update_num_states(self, value):
        self.num_states = value
        self.run_computation(quality='draft')

    def update_grid_points(self, index):
        self.grid_points = self.grid_points_options[index]
        self.run_computation(quality='draft')

    def update_plot_toggle(self, state):
        self.plot_probability_density = self.plot_toggle_checkbox.isChecked()
        self.plot_results()

    def preset_potential_changed(self, index):
        if index == 0:  # Custom drawing
            self.canvas.enabled = True
            self.canvas.clear()
            self.clear_solution()
            self.plot_widget.clear()
            self.plot_widget.setMouseEnabled(x=False, y=False)
        else:
            self.canvas.enabled = False
            self.canvas.clear()
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.generate_preset_potential(index)

    def clear_solution(self):
        attrs = ['x', 'V', 'eigenvalues', 'eigenvectors', 'x_extended']
        for attr in attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def generate_preset_potential(self, index):
        self.clear_solution()
        N = self.grid_points
        potentials = {
            1: ("Square Root |x|", -10, 10, lambda x: np.sqrt(np.abs(x))),
            2: ("Absolute Value |x|", -10, 10, lambda x: np.abs(x)),
            3: ("Harmonic Oscillator x^2", -5, 5, lambda x: x ** 2),
            4: ("Absolute Cubed |x|^3", -4, 4, lambda x: np.abs(x) ** 3 / 2),
            5: ("Quartic x^4", -3, 3, lambda x: x ** 4 / 5),
            6: ("Lennard-Jones Potential", 0.9, 4, lambda x: np.clip(4 * 5 * ((1 / x) ** 12 - (1 / x) ** 6), -20, 50)),
            7: ("Pöschl-Teller Potential", -5, 5, lambda x: -15 / (np.cosh(0.8 * x) ** 2)),
        }
        if index in potentials:
            _, x_min, x_max, func = potentials[index]
            self.x = np.linspace(x_min, x_max, N)
            self.V = func(self.x)
        else:
            return
        self.run_computation(quality='draft')

    def compute_and_plot(self):
        self.run_computation(quality='high')

    def run_computation(self, quality='high'):
        if self.preset_combo.currentIndex() == 0:
            success = self.compute_from_drawing()
        else:
            success = self.load_preset_for_computation()

        if success:
            self.compute_eigenvalues(quality=quality)
            self.plot_results()
        elif self.preset_combo.currentIndex() == 0 and self.sender():
            QMessageBox.warning(self, "Input Error", "Please draw a potential energy curve before computing.")

    def load_preset_for_computation(self):
        return hasattr(self, 'x') and hasattr(self, 'V')

    def compute_from_drawing(self):
        x_data, V_data = self.canvas.get_physical_points()
        if x_data is None or len(x_data) < 4:
            return False

        self.clear_solution()
        self.plot_widget.setMouseEnabled(x=True, y=True)

        sorted_indices = np.argsort(x_data)
        x_data, V_data = x_data[sorted_indices], V_data[sorted_indices]
        unique_indices = np.diff(x_data, prepend=np.nan) != 0
        x_data, V_data = x_data[unique_indices], V_data[unique_indices]

        if np.any(np.diff(x_data) <= 0):
            QMessageBox.warning(self, "Input Error", "The drawn potential must be a single-valued function.")
            return False

        N = self.grid_points
        x_min, x_max = (x_data[0], x_data[-1]) if len(x_data) > 1 else (-10, 10)
        x = np.linspace(x_min, x_max, N)
        try:
            spline = UnivariateSpline(x_data, V_data, s=0, k=3)
            V = spline(x)
        except Exception as e:
            QMessageBox.warning(self, "Computation Error", f"Spline fitting failed: {e}")
            return False

        self.x = x
        self.V = V
        return True

    def extend_potential_if_open(self, x, V, quality):
        extension_factor = 20.0 if quality == 'high' else 4.0
        original_width = x[-1] - x[0]
        if original_width <= 1e-9: return x, V

        num_pts = len(x)
        if num_pts < 10: return x, V

        check_len = max(2, int(0.05 * num_pts))

        dx_left = x[check_len] - x[0]
        dV_left = V[check_len] - V[0]
        slope_left = dV_left / dx_left if dx_left != 0 else 0

        dx_right = x[-1] - x[-check_len - 1]
        dV_right = V[-1] - V[-check_len - 1]
        slope_right = dV_right / dx_right if dx_right != 0 else 0

        V_range = np.max(V) - np.min(V)
        if V_range <= 1e-9: V_range = np.abs(np.mean(V)) if np.abs(np.mean(V)) > 1e-9 else 1.0

        slope_threshold = 0.1 * (V_range / original_width)

        if abs(slope_left) < slope_threshold:
            num_extension_points = int(extension_factor * original_width / (x[1] - x[0]))
            x_left = np.linspace(x[0] - num_extension_points * (x[1] - x[0]), x[0] - (x[1] - x[0]),
                                 num_extension_points)
            V_left = np.full(num_extension_points, V[0])
            x = np.concatenate((x_left, x))
            V = np.concatenate((V_left, V))

        if abs(slope_right) < slope_threshold:
            num_extension_points = int(extension_factor * original_width / (x[-1] - x[-2]))
            x_right = np.linspace(x[-1] + (x[-1] - x[-2]), x[-1] + num_extension_points * (x[-1] - x[-2]),
                                  num_extension_points)
            V_right = np.full(num_extension_points, V[-1])
            x = np.concatenate((x, x_right))
            V = np.concatenate((V, V_right))

        return x, V

    def compute_eigenvalues(self, quality='high'):
        if not hasattr(self, 'x') or not hasattr(self, 'V'): return
        x_extended, V_extended = self.extend_potential_if_open(self.x, self.V, quality)

        N = len(x_extended)
        if N < 3: return
        dx = x_extended[1] - x_extended[0]
        hbar = 1
        coeff = hbar ** 2 / (2 * self.mass * dx ** 2)

        main_diag = np.full(N, 2 * coeff) + V_extended
        off_diag = np.full(N - 1, -coeff)

        H = diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csr')
        H = (H + H.T) / 2

        num_levels = min(self.num_states, N - 2)
        if num_levels <= 0: return
        try:
            eigenvalues, eigenvectors = eigsh(H, k=num_levels, which='SA', tol=1e-6)
        except Exception as e:
            QMessageBox.warning(self, "Computation Error", f"Eigenvalue computation failed: {e}")
            return

        self.x_extended = x_extended
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def plot_results(self):
        self.plot_widget.clear()
        if not hasattr(self, 'x') or not hasattr(self, 'V'): return

        self.plot_widget.plot(self.x, self.V, pen=pg.mkPen('y', width=2), name='Potential Energy V(x)')

        if not hasattr(self, 'eigenvalues'): return

        idx = np.argsort(self.eigenvalues)
        eigenvalues = self.eigenvalues[idx]
        eigenvectors = self.eigenvectors[:, idx]

        y_min, y_max = np.min(self.V), np.max(self.V)

        for E in eigenvalues:
            self.plot_widget.plot([self.x[0], self.x[-1]], [E, E], pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
            y_min = min(y_min, E)
            y_max = max(y_max, E)

        V_range = y_max - y_min if y_max > y_min else 1
        scaling_factor = V_range * self.wavefunc_scale

        for i in range(len(eigenvalues)):
            psi = eigenvectors[:, i]
            energy = eigenvalues[i]

            norm = np.sqrt(np.trapz(psi ** 2, self.x_extended))
            psi_normalized = psi / norm if norm != 0 else psi

            psi_max_abs = np.max(np.abs(psi_normalized))
            psi_scaled = psi_normalized / psi_max_abs if psi_max_abs != 0 else psi_normalized

            if self.plot_probability_density:
                psi_to_plot = np.abs(psi_scaled) ** 2 * scaling_factor
            else:
                psi_to_plot = psi_scaled * scaling_factor

            self.plot_widget.plot(self.x_extended, energy + psi_to_plot, pen=pg.mkPen('g'))
            y_max = max(y_max, np.max(energy + psi_to_plot))

        padding = (y_max - y_min) * 0.1
        self.plot_widget.setYRange(y_min - padding, y_max + padding)
        self.plot_widget.setXRange(np.min(self.x), np.max(self.x))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuantumVisualizer()
    window.show()
    sys.exit(app.exec())
