# quantum_visualizer.py

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QMessageBox, QComboBox, QCheckBox,
    QSpinBox
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QPainterPath
import pyqtgraph as pg
from scipy.interpolate import UnivariateSpline
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setStyleSheet("background-color: white;")
        self.drawing = False
        self.points = []
        self.enabled = True  # To enable or disable drawing

    def mousePressEvent(self, event):
        if self.enabled and event.button() == Qt.LeftButton:
            self.drawing = True
            self.points = [event.pos()]
            self.update()

    def mouseMoveEvent(self, event):
        if self.enabled and self.drawing:
            self.points.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.enabled and event.button() == Qt.LeftButton:
            self.drawing = False
            # Signal that drawing is complete if needed

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.black, 2)
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

        # Normalize x to range [x_min, x_max]
        x_min, x_max = -10, 10  # Adjust as needed
        x_physical = x_pixels / width * (x_max - x_min) + x_min

        # Normalize y to range [V_min, V_max]
        V_min, V_max = 0, 10  # Adjust V_max as needed
        y_physical = (1 - y_pixels / height) * (V_max - V_min) + V_min  # Invert y-axis

        return x_physical, y_physical

    def clear(self):
        self.points = []
        self.update()

class QuantumVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Potential Visualizer")
        self.setGeometry(100, 100, 1200, 600)
        self.mass = 1.0  # Default mass
        self.wavefunc_scale = 0.3  # Default wavefunction scaling factor
        self.plot_probability_density = False  # Default to plotting wavefunctions
        self.num_states = 5  # Default number of states
        self.initUI()

    def initUI(self):
        # Set the background of the plot widget to black
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left side layout (drawing canvas and controls)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Right side layout (plot)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        # Instructions
        instructions = QLabel("Draw a potential energy curve below, select preset potentials, adjust parameters, then click Compute.")
        left_layout.addWidget(instructions)

        # Drawing canvas
        self.canvas = DrawingCanvas()
        left_layout.addWidget(self.canvas)

        # Preset potentials dropdown
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset Potential:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom Drawing")
        self.preset_combo.addItem("Square Root |x|")
        self.preset_combo.addItem("Absolute Value |x|")
        self.preset_combo.addItem("Harmonic Oscillator x^2")
        self.preset_combo.addItem("Absolute Cubed |x|^3")
        self.preset_combo.addItem("Quartic x^4")
        self.preset_combo.addItem("Lennard-Jones Potential")
        self.preset_combo.addItem("Pöschl-Teller Potential")
        self.preset_combo.currentIndexChanged.connect(self.preset_potential_changed)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        left_layout.addLayout(preset_layout)

        # Mass slider layout
        slider_layout = QHBoxLayout()
        mass_label = QLabel("Mass:")
        self.mass_slider = QSlider(Qt.Horizontal)
        self.mass_slider.setMinimum(1)
        self.mass_slider.setMaximum(100)
        self.mass_slider.setValue(10)
        self.mass_slider.valueChanged.connect(self.update_mass)
        slider_layout.addWidget(mass_label)
        slider_layout.addWidget(self.mass_slider)
        left_layout.addLayout(slider_layout)

        # Wavefunction scaling slider
        wf_slider_layout = QHBoxLayout()
        wf_scale_label = QLabel("Wavefunction Scale:")
        self.wf_scale_slider = QSlider(Qt.Horizontal)
        self.wf_scale_slider.setMinimum(1)
        self.wf_scale_slider.setMaximum(100)
        self.wf_scale_slider.setValue(30)
        self.wf_scale_slider.valueChanged.connect(self.update_wf_scale)
        wf_slider_layout.addWidget(wf_scale_label)
        wf_slider_layout.addWidget(self.wf_scale_slider)
        left_layout.addLayout(wf_slider_layout)

        # Number of states control
        num_states_layout = QHBoxLayout()
        num_states_label = QLabel("Number of States:")
        self.num_states_spinbox = QSpinBox()
        self.num_states_spinbox.setMinimum(1)
        self.num_states_spinbox.setMaximum(10)
        self.num_states_spinbox.setValue(self.num_states)
        self.num_states_spinbox.valueChanged.connect(self.update_num_states)
        num_states_layout.addWidget(num_states_label)
        num_states_layout.addWidget(self.num_states_spinbox)
        left_layout.addLayout(num_states_layout)

        # Toggle for plotting wavefunction or probability density
        toggle_layout = QHBoxLayout()
        self.plot_toggle_checkbox = QCheckBox("Plot Probability Density |ψ|²")
        self.plot_toggle_checkbox.stateChanged.connect(self.update_plot_toggle)
        toggle_layout.addWidget(self.plot_toggle_checkbox)
        left_layout.addLayout(toggle_layout)

        # Controls layout
        controls_layout = QHBoxLayout()
        # Compute button
        compute_button = QPushButton("Compute")
        compute_button.clicked.connect(self.compute_and_plot)
        controls_layout.addWidget(compute_button)
        # Clear button
        clear_button = QPushButton("Clear Drawing")
        clear_button.clicked.connect(self.canvas.clear)
        controls_layout.addWidget(clear_button)
        left_layout.addLayout(controls_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Energy')
        self.plot_widget.setLabel('bottom', 'Position')
        self.plot_widget.addLegend()
        right_layout.addWidget(self.plot_widget)

    def update_mass(self, value):
        self.mass = value / 10.0  # Adjust scaling as needed
        # Ensure mass is not zero or too small
        if self.mass < 0.1:
            self.mass = 0.1
        # print(f"Mass set to: {self.mass}")

    def update_wf_scale(self, value):
        self.wavefunc_scale = value / 100.0  # Adjust scaling as needed
        # print(f"Wavefunction scaling factor set to: {self.wavefunc_scale}")

    def update_num_states(self, value):
        self.num_states = value
        # print(f"Number of states set to: {self.num_states}")

    def update_plot_toggle(self, state):
        self.plot_probability_density = self.plot_toggle_checkbox.isChecked()
        # print(f"Plot probability density: {self.plot_probability_density}")

    def preset_potential_changed(self, index):
        if index == 0:
            # Custom drawing
            self.canvas.enabled = True
            self.canvas.clear()
            # Clear stored potential and eigenvalues
            if hasattr(self, 'x'):
                del self.x
            if hasattr(self, 'V'):
                del self.V
            if hasattr(self, 'eigenvalues'):
                del self.eigenvalues
            if hasattr(self, 'eigenvectors'):
                del self.eigenvectors
            self.plot_widget.clear()
        else:
            # Disable drawing and generate preset potential
            self.canvas.enabled = False
            self.canvas.clear()
            self.generate_preset_potential(index)

    def generate_preset_potential(self, index):
        # Adjust x-range and parameters for better scaling
        N = 500

        if index == 1:
            # Square Root |x|
            x_min, x_max = -10, 10
            x = np.linspace(x_min, x_max, N)
            V = np.sqrt(np.abs(x))
        elif index == 2:
            # Absolute Value |x|
            x_min, x_max = -10, 10
            x = np.linspace(x_min, x_max, N)
            V = np.abs(x)
        elif index == 3:
            # Harmonic Oscillator x^2
            x_min, x_max = -5, 5
            x = np.linspace(x_min, x_max, N)
            V = x**2
        elif index == 4:
            # Absolute Cubed |x|^3
            x_min, x_max = -4, 4
            x = np.linspace(x_min, x_max, N)
            V = np.abs(x)**3 / 2  # Adjust scaling
        elif index == 5:
            # Quartic x^4
            x_min, x_max = -3, 3
            x = np.linspace(x_min, x_max, N)
            V = x**4 / 5  # Adjust scaling
        elif index == 6:
            # Lennard-Jones Potential
            x_min, x_max = 0.9, 4
            x = np.linspace(x_min, x_max, N)
            epsilon = 5  # Adjusted depth
            sigma = 1
            V = 4 * epsilon * ((sigma / x)**12 - (sigma / x)**6)
            V = np.clip(V, -20, 50)  # Adjust clipping
        elif index == 7:
            # Pöschl-Teller Potential
            x_min, x_max = -5, 5
            x = np.linspace(x_min, x_max, N)
            V0 = 15
            a = 0.8
            V = -V0 / (np.cosh(a * x)**2)
        else:
            x_min, x_max = -10, 10
            x = np.linspace(x_min, x_max, N)
            V = np.zeros_like(x)

        # Store the potential and x for computation
        self.x = x
        self.V = V

        # Clear any previous eigenvalues and eigenvectors
        if hasattr(self, 'eigenvalues'):
            del self.eigenvalues
        if hasattr(self, 'eigenvectors'):
            del self.eigenvectors

        # Update the plot
        self.plot_results()

    def compute_and_plot(self):
        if self.preset_combo.currentIndex() == 0:
            # Use custom drawing
            self.compute_solution()
        else:
            # Compute eigenvalues for preset potential
            self.compute_eigenvalues()
        self.plot_results()

    def compute_solution(self):
        # Get the physical points from the canvas
        x_data, V_data = self.canvas.get_physical_points()
        if x_data is None or len(x_data) < 4:
            QMessageBox.warning(self, "Input Error", "Please draw a potential energy curve with enough points.")
            return

        # Convert to numpy arrays
        x_data = np.array(x_data)
        V_data = np.array(V_data)

        # Sort the data by x to ensure a proper function
        sorted_indices = np.argsort(x_data)
        x_data = x_data[sorted_indices]
        V_data = V_data[sorted_indices]

        # Remove duplicate x values
        unique_indices = np.diff(x_data, prepend=np.nan) != 0
        x_data = x_data[unique_indices]
        V_data = V_data[unique_indices]

        # Check if x_data is strictly increasing
        if np.any(np.diff(x_data) <= 0):
            QMessageBox.warning(self, "Input Error", "The drawn potential must be a single-valued function.")
            return

        # Create a uniform spatial grid
        N = 500  # Number of spatial points
        x_min, x_max = x_data[0], x_data[-1]
        x = np.linspace(x_min, x_max, N)

        # Spline fit to get V(x)
        try:
            spline = UnivariateSpline(x_data, V_data, s=0, k=3)
            V = spline(x)
        except Exception as e:
            QMessageBox.warning(self, "Computation Error", f"Spline fitting failed: {e}")
            return

        # Store results for plotting potential
        self.x = x
        self.V = V

        # Check for NaNs or infinities in V
        if np.any(np.isnan(V)) or np.any(np.isinf(V)):
            QMessageBox.warning(self, "Computation Error", "Potential V(x) contains NaNs or infinities.")
            return

        self.compute_eigenvalues()

    def compute_eigenvalues(self):
        x = self.x
        V = self.V
        N = len(x)

        # Mass from the slider
        m = self.mass
        if m < 0.1:
            QMessageBox.warning(self, "Input Error", "Mass is too small for stable computation.")
            return

        # Set up the Hamiltonian
        dx = x[1] - x[0]
        hbar = 1  # Planck constant (set to 1 for simplicity)
        coeff = hbar**2 / (2 * m * dx**2)

        # Kinetic energy matrix
        main_diag = np.full(N, 2 * coeff)
        off_diag = np.full(N - 1, -coeff)

        # Potential energy
        V_diag = V

        # Hamiltonian matrix
        diagonals = [main_diag + V_diag, off_diag, off_diag]
        H = diags(diagonals, [0, -1, 1], format='csr')

        # Ensure H is symmetric
        H = (H + H.getH()) / 2

        # Solve eigenvalue problem
        num_levels = min(self.num_states, N - 2)
        try:
            eigenvalues, eigenvectors = eigsh(H, k=num_levels, which='SA', tol=1e-6, maxiter=1000)
        except Exception as e:
            QMessageBox.warning(self, "Computation Error", f"Eigenvalue computation failed: {e}")
            return

        # Store results for plotting
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def plot_results(self):
        if not hasattr(self, 'x') or not hasattr(self, 'V'):
            return

        self.plot_widget.clear()

        # Plot potential as a smooth curve
        self.plot_widget.plot(self.x, self.V, pen=pg.mkPen('y', width=2), name='Potential Energy')

        # Initialize Y-axis limits
        y_min = np.min(self.V)
        y_max = np.max(self.V)

        # Plot energy levels and wavefunctions if available
        if hasattr(self, 'eigenvalues') and hasattr(self, 'eigenvectors'):
            # Sort eigenvalues and eigenvectors
            idx = np.argsort(self.eigenvalues)
            eigenvalues = self.eigenvalues[idx]
            eigenvectors = self.eigenvectors[:, idx]

            # Plot energy levels
            for E in eigenvalues:
                self.plot_widget.addLine(y=E, pen=pg.mkPen('r', style=Qt.DashLine))
                # Update Y-axis limits based on energy levels
                y_min = min(y_min, E)
                y_max = max(y_max, E)

            # Calculate scaling factor based on potential range
            V_range = y_max - y_min if y_max != y_min else 1  # Prevent division by zero
            scaling_factor = V_range * self.wavefunc_scale  # Use the wavefunction scaling factor from slider

            # Plot wavefunctions offset by their energy levels
            for i in range(len(eigenvalues)):
                psi = eigenvectors[:, i]
                # Normalize the wavefunction
                norm = np.sqrt(np.trapz(psi**2, self.x))
                if norm == 0:
                    norm = 1  # Prevent division by zero
                psi_normalized = psi / norm
                # Scale the wavefunction
                psi_scaled = psi_normalized / np.max(np.abs(psi_normalized))

                energy = eigenvalues[i]

                if self.plot_probability_density:
                    # Plot probability density |ψ|^2
                    psi_to_plot = np.abs(psi_scaled)**2 * scaling_factor
                else:
                    # Plot wavefunction ψ
                    psi_to_plot = psi_scaled * scaling_factor

                self.plot_widget.plot(self.x, energy + psi_to_plot, pen=pg.mkPen('g'))
                # Update Y-axis limits based on wavefunctions
                y_min = min(y_min, np.min(energy + psi_to_plot))
                y_max = max(y_max, np.max(energy + psi_to_plot))

        # Adjust plot ranges
        padding = (y_max - y_min) * 0.1  # Add 10% padding
        self.plot_widget.setYRange(y_min - padding, y_max + padding)
        self.plot_widget.setXRange(np.min(self.x), np.max(self.x))

    def closeEvent(self, event):
        # Optional: Add any cleanup code here
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuantumVisualizer()
    window.show()
    sys.exit(app.exec_())
