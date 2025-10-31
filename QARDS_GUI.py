# THIS IS A GUI VERSION, REPLACES CLI INPUTS
from __future__ import annotations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, random_unitary
import numpy as np
import random
import sys

# Qt + Matplotlib
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QMessageBox, QDialogButtonBox, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mplcursors

# Application icon
ICON_PATH_LIGHT = "qards_icon_light.svg"
APP_ICON_LIGHT  = QtGui.QIcon(ICON_PATH_LIGHT)
ICON_PATH_DARK  = "qards_icon_dark.svg"
APP_ICON_DARK   = QtGui.QIcon(ICON_PATH_DARK)

try:
  UNIQUE = QtCore.Qt.ConnectionType.UniqueConnection
except AttributeError:
  UNIQUE = QtCore.Qt.UniqueConnection

# User options
enable_skip = False
enable_measure = False

"""
Quantum cards with measurement
1. There are n (3 to 7, inclusive) qubits, starting in a random state
2. Each player is assigned bitstrings of length n, which, if it is a measurement
outcome, earns them points.
Player with the most points after running 1024 shots of measurements wins.
3. Each player has some cards that correspond to unitary gates
They may pick which qubits it operates on, however
4. Special cards besides the standard unitary gates we learned in class:
Diffusion operator card: applies the diffusion operator from Grover's which can
help if they have a good chance of their bitstring being measured
Reverse card: Undo the other player's last move
Reset a qubit to zero

5. Once there are no more cards, measure the outcome and assign points.

How this meets the requirements:
1. Superposition: The qubits begin in a random state and the quantum state is
a superposition
2. Entanglement: Can be achieved through controlled gates. Useful for Grover
oracles and useful strategically if one has strings like 000 and 110 where
one can use CNOT to entangle the 2 leftmost qubits.
3. Measurement: obvious
4. Interference: Amplitude amplification/Grover, terms can cancel in the math
of the game plausibly
"""

MOVE_NAMES = [
    'X', 'Y', 'Z', 'H', 'CX', 'CCX', 'I', 'S', 'S_dag', 'T', 'T_dag',
    'SWAP', 'DIFFUSION', 'CZ', 'CCZ', 'RESET', 'GROVER', 'REVERSE'
]

MOVE_WEIGHTS = [
    4, 1, 3, 5, 3, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2
]

MOVE_TO_NUM_QUBITS = {
    'X': 1, 'Y': 1, 'Z': 1, 'H': 1, 'CX': 2, 'CCX': 3, 'I': 0, 'S': 1, 'S_dag': 1,
    'T': 1, 'T_dag': 1, 'SWAP': 2, 'CZ': 2, 'CCZ': 3, 'DIFFUSION': 0, 'RESET': 1,
    'GROVER': 0, 'REVERSE': 0
}

def generate_bitstring(length: int):
  """Generates a random bitstring of length n. Used for generating strings
  at the beginning of the game for each player to aim for as a measurement
  outcome."""
  bitstring = ""
  for _ in range(length):
    bitstring += random.choice(['0', '1'])
  return bitstring


class Card:
  """Class storing information about a card: operation and number of qubits"""
  def __init__(self, operation_name, num_qubits_involved):
    self.operation_name = operation_name
    self.num_qubits_involved = num_qubits_involved

  def __repr__(self):
    return self.operation_name


def generate_random_deck(num_cards, num_qubits_in_game):
  """Generates a random deck at the beginning of the game"""
  card_names = random.choices(MOVE_NAMES, MOVE_WEIGHTS, k=num_cards)
  while num_cards == 1 and (card_names[0] == 'REVERSE' or card_names[0] == 'I'):
    # Avoid single identity or reverse card decks
    card_names = random.choices(MOVE_NAMES, MOVE_WEIGHTS, k=num_cards)
  cards = [Card(card_name, MOVE_TO_NUM_QUBITS.get(card_name, num_qubits_in_game)) for card_name in card_names]
  return cards


def plot_statevector_data(statevector_dict: dict[str, float], ax, fig) -> None:
  """
  Plots a statevector probability dict onto the provided Matplotlib axes.
  Keys are bitstrings; values are probabilities.
  """
  ax.clear()

  keys = sorted(statevector_dict.keys())
  values = [statevector_dict[k] for k in keys]

  x = np.arange(len(keys))
  bars = ax.bar(x, values, width=0.5)
  ax.set_xticks(x)
  ax.set_xticklabels(keys, rotation=45)
  ax.set_ylabel('|Probability = Amplitude|^2')
  ax.set_xlabel('Measurement Outcome')
  ax.set_ylim(0, 1)

  fig.canvas.draw_idle()


def plot_counts(counts: dict[str, int], ax, fig, thresh_to_plot: float = 0.00, normalize: bool = False) -> None:
  """
  Plots measurement counts onto the provided Matplotlib axes.
  If normalize=True, values are fractions of total shots.
  """
  total = sum(counts.values()) if counts else 0
  if total == 0:
    ax.clear()
    ax.set_title('No counts to plot')
    fig.canvas.draw_idle()
    return

  items = [(k, (v/total) if normalize else v) for k, v in counts.items()]
  if normalize:
    items = [(k, v) for k, v in items if v >= thresh_to_plot]

  items.sort(key=lambda kv: int(kv[0], 2))
  keys = [k for k, _ in items]
  vals = [v for _, v in items]

  ax.clear()
  x = np.arange(len(keys))
  bars = ax.bar(x, vals, width=0.5)

  ax.set_xlabel('Measurement Outcome')
  if normalize:
    ax.set_ylabel('Fraction of shots')
    ymax = max(vals) if vals else 1
    ax.set_ylim(0, max(1.0, ymax * 1.15))
  else:
    ax.set_ylabel('Counts')
    ymax = max(vals) if vals else 1
    ax.set_ylim(0, ymax * 1.15)

  ax.set_xticks(x)
  ax.set_xticklabels(keys, rotation=45, ha='right')

  for rect, v in zip(bars, vals):
    height = rect.get_height()
    label = f"{v:.3f}" if normalize else f"{int(v)}"
    ax.annotate(label,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom')

  fig.canvas.draw_idle()


class Game:
  """Class that manages game loop and flow"""

  def __init__(self, num_qubits, num_cards, num_bitstrings_per_player=0):
    """
    Constructor for Game object
    :param num_qubits: Number of qubits in the circuit for the game
    :type num_qubits: int
    :param num_cards: Number of cards in each player's deck to begin the game
    :type num_cards: int
    :param num_bitstrings_per_player: Number of target bitstrings for each player
    :type num_bitstrings_per_player: int
    """
    if num_bitstrings_per_player == 0:
      num_bitstrings_per_player = int(2**(num_qubits - 2))

    # Initialize random quantum circuit
    self.num_qubits = num_qubits
    self.q: QuantumRegister = QuantumRegister(num_qubits)
    self.c: ClassicalRegister = ClassicalRegister(num_qubits)
    self.qc: QuantumCircuit = QuantumCircuit(self.q, self.c)
    U = random_unitary(2**self.num_qubits)
    self.qc.unitary(U, self.q)

    # Generate decks
    self.num_cards_per_player = num_cards
    self.deck_player_one = generate_random_deck(num_cards, num_qubits)
    self.deck_player_two = generate_random_deck(num_cards, num_qubits)

    # Assign target bitstrings
    self.num_target_bitstrings = num_bitstrings_per_player
    self.bitstrings_player_one = set()
    while len(self.bitstrings_player_one) < self.num_target_bitstrings:
      self.bitstrings_player_one.add(generate_bitstring(self.num_qubits))
    self.bitstrings_player_two = set()
    while len(self.bitstrings_player_two) < self.num_target_bitstrings:
      the_random_bitstring = generate_bitstring(self.num_qubits)
      if the_random_bitstring in self.bitstrings_player_one:
        continue
      self.bitstrings_player_two.add(the_random_bitstring)

    # Decide who goes first
    self.turn = random.choice([1, 2])
    self.list_of_previous_circuits = []

  def apply_grover_oracle_with_planted_sol(self, bitstring: str):
    """
    Applies a Grover oracle with the planted solution 
    corresponding to bitstring
    :param bitstring: The bitstring to perform an iteration of Grover for
    """
    for i in range(len(bitstring)):
        index = len(bitstring) - 1 - i
        if bitstring[index] == '0':
            self.qc.x(self.q[index])
    self.qc.mcp(np.pi,self.q[1:],self.q[0])
    for i in range(len(bitstring)):
        index = len(bitstring) - 1 - i
        if bitstring[index] == '0':
            self.qc.x(self.q[index])

  def diffusion_operator(self):
    self.qc.h(self.q)
    self.qc.x(self.q)
    self.qc.mcp(np.pi,self.q[1:],self.q[0])
    self.qc.x(self.q)
    self.qc.h(self.q)

  def apply_move(self, card_played: Card, qubit_numbers=None, grover_string=None):
    """
    Applies the selected card/gate to the circuit.
    """
    op = card_played.operation_name
    qubit_numbers = qubit_numbers or []

    # REVERSE is special: restore last snapshot
    if op == 'REVERSE':
      if self.list_of_previous_circuits:
        self.qc = self.list_of_previous_circuits.pop()
      # If no history, do nothing (no-op)
      return

    self.list_of_previous_circuits.append(self.qc.copy())
    self.qc = self.qc.copy()
    q = self.q

    match op:
      case 'X':
        self.qc.x(q[qubit_numbers[0]])
      case 'Y':
        self.qc.y(q[qubit_numbers[0]])
      case 'Z':
        self.qc.z(q[qubit_numbers[0]])
      case 'H':
        self.qc.h(q[qubit_numbers[0]])
      case 'CX':
        self.qc.cx(q[qubit_numbers[0]], q[qubit_numbers[1]])
      case 'CZ':
        self.qc.cz(q[qubit_numbers[0]], q[qubit_numbers[1]])
      case 'CCX':
        self.qc.ccx(q[qubit_numbers[0]], q[qubit_numbers[1]], q[qubit_numbers[2]])
      case 'CCZ':
        self.qc.ccz(q[qubit_numbers[0]], q[qubit_numbers[1]], q[qubit_numbers[2]])
      case 'S':
        self.qc.s(q[qubit_numbers[0]])
      case 'S_dag':
        self.qc.sdg(q[qubit_numbers[0]])
      case 'T':
        self.qc.t(q[qubit_numbers[0]])
      case 'T_dag':
        self.qc.tdg(q[qubit_numbers[0]])
      case 'SWAP':
        self.qc.swap(q[qubit_numbers[0]], q[qubit_numbers[1]])
      case 'DIFFUSION':
        self.diffusion_operator()
      case 'RESET':
        self.qc.reset(q[qubit_numbers[0]])
        # Randomize post-reset |0> -> |1> half the time
        if random.random() < 0.5:
          self.qc.x(q[qubit_numbers[0]])
      case 'GROVER':
        # Validate bitstring length and contents
        if (not grover_string or
            len(grover_string) != self.num_qubits or
            any(ch not in '01' for ch in grover_string)):
          raise ValueError("Invalid GROVER bitstring.")
        self.apply_grover_oracle_with_planted_sol(grover_string)
        self.diffusion_operator()
      case 'I':
        pass

  def measure_output_end_of_game(self):
    """Measures output at the end of the game"""
    qc_m = self.qc.copy()
    for i in range(self.num_qubits):
      qc_m.measure(self.q[i], self.c[i])
    backend = Aer.get_backend('qasm_simulator')
    NUM_SHOTS = 1024
    result = backend.run(qc_m, shots=NUM_SHOTS).result()
    counts = result.get_counts()
    return counts

  def get_statevector_data(self):
    """Gets the statevector data after all the moves that have been applied
    thus far"""
    v_init = Statevector.from_label('0' * self.num_qubits)
    v_evolved = v_init.evolve(self.qc)
    return v_evolved.probabilities_dict()

  def get_target_bitstring_counts(self, counts):
    """
    Determines number of times that each player has one of their target
    bitstrings occur.
    :param counts: Number of occurrences of each measured outcome
    :type counts: dict
    :returns: tuple(int, int) with counts for each player
    """
    count_player_one = 0
    count_player_two = 0
    for key in counts:
      if key in self.bitstrings_player_one:
        count_player_one += counts[key]
      if key in self.bitstrings_player_two:
        count_player_two += counts[key]
    return (count_player_one, count_player_two)

  def show_game_rules(self, parent=None):
    """Show the game rules in a popup message box."""
    rules = f"""
    <b>QARDS: Quantum Card Game</b><br><br>
    <b>QARDS: The Quantum Card Game</b><br><br>

    <b>1.</b> There are <b>{self.num_qubits}</b> qubits, starting in a random (possibly entangled) state.<br>
    <b>2.</b> Each player has <b>{self.num_target_bitstrings}</b> target bitstrings of length {self.num_qubits}.
    If one of these is measured, that player earns a point.<br>
    <b>3.</b> Each player has <b>{self.num_cards_per_player}</b> cards that apply quantum operations.<br>
    <b>4.</b> Players alternate turns playing cards. When both decks are empty, the qubits are measured, and the player whose bitstrings appear most often wins.<br><br>

    <b>Card Reference:</b><br>
    - X, Y, Z, H, CX, CCX, CZ, CCZ, S, T, S_dag, T_dag, I, SWAP:<br>
      Apply the corresponding quantum gate on chosen qubit(s).<br>
    - DIFFUSION: Grover diffusion operator.<br>
    - RESET: Resets a qubit to |0⟩ (randomly flips to |1⟩ half the time).<br>
    - GROVER: Runs a Grover iteration amplifying a bitstring chosen by the player.<br>
    - REVERSE: Undoes the previous move (restores the prior circuit).<br><br>

    <i>Note:</i> Bitstrings follow little-endian order (Qubit 0 is rightmost).
    """

    msg = QMessageBox()
    msg.setWindowTitle("Game Rules")
    msg.setTextFormat(Qt.TextFormat.RichText)
    msg.setText(rules)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


# Embedded Matplotlib canvases for GUI
class MplCanvas(FigureCanvas):
  """Small helper to host a Matplotlib figure as a Qt widget"""
  def __init__(self, title: str):
    self.fig = Figure(figsize=(6, 4), tight_layout=True)
    super().__init__(self.fig)
    self.ax = self.fig.add_subplot(111)
    self.fig.suptitle(title)


# List widget for decks
class DeckView(QtWidgets.QGroupBox):
  """List UI for a player's deck (select a card by index)."""
  selectionChanged = QtCore.Signal(int)

  def __init__(self, title: str):
    super().__init__(title)
    self.list = QtWidgets.QListWidget()
    self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    self.list.currentRowChanged.connect(self.selectionChanged)
    lay = QtWidgets.QVBoxLayout(self)
    lay.addWidget(self.list)

  def populate(self, deck: list[Card]):
    self.list.clear()
    for i, card in enumerate(deck):
      self.list.addItem(f"{i}: {card.operation_name}")

  def selected_row(self) -> int:
    return self.list.currentRow()

  def remove_row(self, row: int):
    it = self.list.takeItem(row)
    if it:
      del it

  def set_enabled_for_turn(self, enabled: bool):
    base = self.title().split(' [')[0]
    self.setTitle(base + (" [your turn]" if enabled else ""))
    self.list.setEnabled(enabled)


class GameSetupDialog(QDialog):
    """Dialog for selecting the number of qubits, cards, and bitstrings per player."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game Setup")

        layout = QVBoxLayout(self)

        # Number of qubits
        qubit_layout = QHBoxLayout()
        qubit_layout.addWidget(QLabel("Number of qubits (3-5):"))
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setRange(3, 5)
        self.qubit_spin.setValue(3)
        qubit_layout.addWidget(self.qubit_spin)
        layout.addLayout(qubit_layout)

        # Number of cards
        cards_layout = QHBoxLayout()
        cards_layout.addWidget(QLabel("Cards per player (1-20):"))
        self.cards_spin = QSpinBox()
        self.cards_spin.setRange(1, 20)
        self.cards_spin.setValue(5)
        cards_layout.addWidget(self.cards_spin)
        layout.addLayout(cards_layout)

        # Number of bitstrings
        bits_layout = QHBoxLayout()
        bits_layout.addWidget(QLabel("Bitstrings per player (0 = default):"))
        self.bits_spin = QSpinBox()
        self.bits_spin.setRange(0, 2 ** (self.qubit_spin.value() - 1))
        self.bits_spin.setValue(0)
        bits_layout.addWidget(self.bits_spin)
        layout.addLayout(bits_layout)
        self.qubit_spin.valueChanged.connect(self.update_bitstring_limit)

        self.early_measure_checkbox = QCheckBox("Enable early measurement (before all cards are played)")
        self.early_measure_checkbox.setChecked(False)
        layout.addWidget(self.early_measure_checkbox)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def update_bitstring_limit(self, new_qubit_value: int):
        """Update max allowed bitstrings when qubit count changes."""
        max_bits = 2 ** (new_qubit_value - 1)
        self.bits_spin.setMaximum(max_bits)
        # Optionally adjust current value if it's now above the limit
        if self.bits_spin.value() > max_bits:
            self.bits_spin.setValue(max_bits)

    def get_values(self):
        return (
            self.qubit_spin.value(),
            self.cards_spin.value(),
            self.bits_spin.value(),
            self.early_measure_checkbox.isChecked()
        )


class MainWindow(QtWidgets.QWidget):
  """Class that manages GUI loop and flow"""

  def __init__(self):
    super().__init__()
    self.setWindowTitle("QARDS - GUI")
    self.resize(1300, 820)

    setup_dialog = GameSetupDialog(self)
    if setup_dialog.exec() == QDialog.Accepted:
        num_qubits, num_cards, num_bitstrings, enable_measure = setup_dialog.get_values()
        if num_bitstrings == 0 or num_bitstrings > int(2 ** (num_qubits - 1)):
            num_bitstrings = int(2 ** (num_qubits - 2))
    else:
        sys.exit(0)

    self.game = Game(num_qubits, num_cards, num_bitstrings)
    self.enable_measure = enable_measure
    self.game.show_game_rules(parent=self)

    # Decks + targets
    left = QtWidgets.QVBoxLayout()
    self.deck_p1 = DeckView("Player 1 Deck")
    self.deck_p2 = DeckView("Player 2 Deck")
    left.addWidget(self.deck_p1)
    left.addWidget(self.deck_p2)
    self.targets_p1 = QtWidgets.QLabel()
    self.targets_p2 = QtWidgets.QLabel()
    left.addWidget(QtWidgets.QLabel("Player 1 bitstrings:"))
    left.addWidget(self.targets_p1)
    left.addWidget(QtWidgets.QLabel("Player 2 bitstrings:"))
    left.addWidget(self.targets_p2)
    left.addStretch(1)

    # Move controls
    mid = QtWidgets.QVBoxLayout()
    self.turn_label = QtWidgets.QLabel()
    self.turn_label.setStyleSheet("font-weight: 600; font-size: 16px;")
    mid.addWidget(self.turn_label)

    self.simple_selector_box = QtWidgets.QGroupBox("Select qubits for the selected card:")
    row = QtWidgets.QHBoxLayout(self.simple_selector_box)
    self.qchecks: list[QtWidgets.QCheckBox] = []
    for i in range(self.game.num_qubits):
      cb = QtWidgets.QCheckBox(str(i))
      row.addWidget(cb)
      self.qchecks.append(cb)
    mid.addWidget(self.simple_selector_box)

    # Buttons
    row_btns = QtWidgets.QHBoxLayout()
    self.play_btn = QtWidgets.QPushButton("Play Selected Card")
    row_btns.addWidget(self.play_btn)    
    self.skip_btn = QtWidgets.QPushButton("End Turn (No Move)")
    if (enable_skip):
      row_btns.addWidget(self.skip_btn)
    self.measure_btn = QtWidgets.QPushButton("Measure / End Game")
    if (enable_measure):
      row_btns.addWidget(self.measure_btn)
    mid.addLayout(row_btns)

    self.selector_box = QtWidgets.QGroupBox("Gate qubit selection")
    sel_lay = QtWidgets.QFormLayout(self.selector_box)

    self.ctrl1_combo = QtWidgets.QComboBox()
    self.ctrl2_combo = QtWidgets.QComboBox()
    self.tgt_combo   = QtWidgets.QComboBox()
    for i in range(self.game.num_qubits):
      s = str(i)
      self.ctrl1_combo.addItem(s)
      self.ctrl2_combo.addItem(s)
      self.tgt_combo.addItem(s)

    self.lbl_ctrl  = QtWidgets.QLabel("Control:")
    self.lbl_ctrl2 = QtWidgets.QLabel("Control 2:")
    self.lbl_tgt   = QtWidgets.QLabel("Target:")

    sel_lay.addRow(self.lbl_ctrl,  self.ctrl1_combo)
    sel_lay.addRow(self.lbl_ctrl2, self.ctrl2_combo)
    sel_lay.addRow(self.lbl_tgt,   self.tgt_combo)

    self.selector_box.setVisible(False)
    self.lbl_ctrl.setVisible(False);  self.ctrl1_combo.setVisible(False)
    self.lbl_ctrl2.setVisible(False); self.ctrl2_combo.setVisible(False)
    self.lbl_tgt.setVisible(False);   self.tgt_combo.setVisible(False)

    mid.addWidget(self.selector_box)
    self.status = QtWidgets.QLabel()
    mid.addWidget(self.status)
    mid.addStretch(1)

    # Embedded plots
    right = QtWidgets.QVBoxLayout()
    self.state_canvas = MplCanvas("Statevector Probabilities")
    self.counts_canvas = MplCanvas("Measurement Results")
    right.addWidget(self.state_canvas, 1)
    right.addWidget(self.counts_canvas, 1)

    # Root layout
    root = QtWidgets.QHBoxLayout(self)
    root.addLayout(left, 1)
    root.addLayout(mid, 1)
    root.addLayout(right, 2)

    self.play_btn.clicked.connect(self.on_play)
    self.skip_btn.clicked.connect(self.on_skip)
    self.measure_btn.clicked.connect(self.on_measure)    
    self.deck_p1.selectionChanged.connect(lambda _: self.on_card_highlight(1))
    self.deck_p2.selectionChanged.connect(lambda _: self.on_card_highlight(2))

    # Auto-clear counts
    self.counts_timer = QtCore.QTimer(self)
    self.counts_timer.setSingleShot(True)
    self.counts_timer.timeout.connect(self.clear_counts_plot)

    self.refresh_all()

  # Helper functions
  def current_player(self) -> int:
    return self.game.turn

  def active_deck(self) -> list[Card]:
    return self.game.deck_player_one if self.current_player() == 1 else self.game.deck_player_two

  def active_deck_view(self) -> DeckView:
    return self.deck_p1 if self.current_player() == 1 else self.deck_p2

  def selected_qubits(self) -> list[int]:
    return [i for i, cb in enumerate(self.qchecks) if cb.isChecked()]

  def set_status(self, msg: str):
    self.status.setText(msg)

  def refresh_targets(self):
    self.targets_p1.setText(', '.join(sorted(self.game.bitstrings_player_one)))
    self.targets_p2.setText(', '.join(sorted(self.game.bitstrings_player_two)))

  def refresh_decks(self):
    self.deck_p1.populate(self.game.deck_player_one)
    self.deck_p2.populate(self.game.deck_player_two)
    self.deck_p1.set_enabled_for_turn(self.current_player() == 1)
    self.deck_p2.set_enabled_for_turn(self.current_player() == 2)

  def refresh_turn_label(self):
    self.turn_label.setText(f"Current player: Player {self.current_player()}")

  def refresh_state_plot(self):
    probs = self.game.get_statevector_data()
    plot_statevector_data(probs, ax=self.state_canvas.ax, fig=self.state_canvas.fig)

  def clear_counts_plot(self):
    self.counts_canvas.ax.clear()
    self.counts_canvas.fig.canvas.draw_idle()
    self.set_status("Counts cleared.")

  def refresh_all(self):
    self.refresh_turn_label()
    self.refresh_decks()
    self.refresh_targets()
    self.refresh_state_plot()
    self.clear_counts_plot()

  def next_turn(self):
    self.game.turn = 2 if self.game.turn == 1 else 1
    for cb in self.qchecks:
      cb.setChecked(False)
    self.refresh_all()

  def update_measure_enabled(self):
    can_measure = (len(self.game.deck_player_one) == 0
                   and len(self.game.deck_player_two) == 0)
    self.measure_btn.setEnabled(can_measure)
    self.measure_btn.setToolTip("" if can_measure else
        "Play all cards from both decks to enable measuring.")

  def decks_empty(self) -> bool:
    return len(self.game.deck_player_one) == 0 and len(self.game.deck_player_two) == 0

  def lock_after_end(self):
    # Disable inputs after game end
    self.play_btn.setEnabled(False)
    if hasattr(self, "skip_btn"):
        self.skip_btn.setEnabled(False)
    if hasattr(self, "measure_btn"):
        self.measure_btn.setEnabled(False)
    self.deck_p1.list.setEnabled(False)
    self.deck_p2.list.setEnabled(False)
    self.simple_selector_box.setEnabled(False)
    self.selector_box.setEnabled(False)

  def ensure_distinct_selector(self):
    """
    Keep selector combos distinct for CX/CZ (2), CCX/CCZ (3), and SWAP (2).
    Auto-bumps later fields to the next free index when collisions occur.
    """
    if not self.selector_box.isVisible():
      return

    title = self.selector_box.title()
    n = self.game.num_qubits
    if n <= 1:
      return

    def set_idx(combo, idx):
      combo.blockSignals(True)
      combo.setCurrentIndex(idx % n)
      combo.blockSignals(False)

    if title.startswith("CX") or title.startswith("CZ"):
      a = self.ctrl1_combo.currentIndex()
      b = self.tgt_combo.currentIndex()
      if a == b:
        set_idx(self.tgt_combo, (b + 1) % n)

    elif title.startswith("CCX") or title.startswith("CCZ"):
      a = self.ctrl1_combo.currentIndex()
      b = self.ctrl2_combo.currentIndex()
      c = self.tgt_combo.currentIndex()

      used = set()
      for combo, idx in ((self.ctrl1_combo, a), (self.ctrl2_combo, b), (self.tgt_combo, c)):
        cur = idx
        while cur in used and n > len(used):
          cur = (cur + 1) % n
        if cur != idx:
          set_idx(combo, cur)
        used.add(cur)

    elif title.startswith("SWAP"):
      a = self.ctrl1_combo.currentIndex()
      b = self.tgt_combo.currentIndex()
      if a == b:
        set_idx(self.tgt_combo, (b + 1) % n)

  def on_card_highlight(self, which_deck: int):
    deck = self.game.deck_player_one if which_deck == 1 else self.game.deck_player_two
    view = self.deck_p1 if which_deck == 1 else self.deck_p2
    row = view.selected_row()
    op = deck[row].operation_name if 0 <= row < len(deck) else None

    self.selector_box.setVisible(False)
    self.lbl_ctrl.setVisible(False);  self.ctrl1_combo.setVisible(False)
    self.lbl_ctrl2.setVisible(False); self.ctrl2_combo.setVisible(False)
    self.lbl_tgt.setVisible(False);   self.tgt_combo.setVisible(False)

    self.lbl_ctrl.setText("Control:")
    self.lbl_tgt.setText("Target:")

    if not op:
      self.simple_selector_box.setVisible(True)
      return

    if op in ('CX', 'CZ', 'CCX', 'CCZ', 'SWAP'):
      self.simple_selector_box.setVisible(False)
      n = self.game.num_qubits

      def set_distinct_defaults(pairs):
        used = set()
        for combo, start in pairs:
          idx = start % max(1, n)
          while n > 1 and idx in used:
            idx = (idx + 1) % n
          combo.setCurrentIndex(idx)
          used.add(idx)

      if op in ('CX', 'CZ'):
        self.selector_box.setTitle(f"{op} selection")
        self.selector_box.setVisible(True)
        self.lbl_ctrl.setVisible(True); self.ctrl1_combo.setVisible(True)
        self.lbl_tgt.setVisible(True);  self.tgt_combo.setVisible(True)
        set_distinct_defaults([(self.ctrl1_combo, 0), (self.tgt_combo, 1)])
        self.ctrl1_combo.currentIndexChanged.connect(self.ensure_distinct_selector, UNIQUE)
        self.tgt_combo.currentIndexChanged.connect(self.ensure_distinct_selector, UNIQUE)

      elif op in ('CCX', 'CCZ'):
        self.selector_box.setTitle(f"{op} selection")
        self.selector_box.setVisible(True)
        self.lbl_ctrl.setVisible(True);  self.ctrl1_combo.setVisible(True)
        self.lbl_ctrl2.setVisible(True); self.ctrl2_combo.setVisible(True)
        self.lbl_tgt.setVisible(True);   self.tgt_combo.setVisible(True)
        set_distinct_defaults([
          (self.ctrl1_combo, 0),
          (self.ctrl2_combo, 1),
          (self.tgt_combo,   2),
        ])
        for combo in (self.ctrl1_combo, self.ctrl2_combo, self.tgt_combo):
          combo.currentIndexChanged.connect(self.ensure_distinct_selector, UNIQUE)

      elif op == 'SWAP':
        self.selector_box.setTitle("SWAP selection")
        self.selector_box.setVisible(True)
        self.lbl_ctrl.setText("Qubit A:")
        self.lbl_tgt.setText("Qubit B:")
        self.lbl_ctrl.setVisible(True); self.ctrl1_combo.setVisible(True)
        self.lbl_tgt.setVisible(True);  self.tgt_combo.setVisible(True)
        set_distinct_defaults([(self.ctrl1_combo, 0), (self.tgt_combo, 1)])
        self.ctrl1_combo.currentIndexChanged.connect(self.ensure_distinct_selector, UNIQUE)
        self.tgt_combo.currentIndexChanged.connect(self.ensure_distinct_selector, UNIQUE)

    else:
      needs = MOVE_TO_NUM_QUBITS.get(op, 0)
      self.selector_box.setVisible(False)
      self.simple_selector_box.setVisible(needs >= 1)

  # Actions
  def on_play(self):
    deck = self.active_deck()
    view = self.active_deck_view()
    row = view.selected_row()
    if row < 0 or row >= len(deck):
      self.set_status("Select a card in your deck first.")
      return

    card = deck[row]
    if card.operation_name in ('CX', 'CZ', 'CCX', 'CCZ', 'SWAP'):
      self.simple_selector_box.setVisible(False)
    need = MOVE_TO_NUM_QUBITS.get(card.operation_name, 0)

    # CX / CZ
    if card.operation_name in ('CX', 'CZ'):
      a = int(self.ctrl1_combo.currentText())
      b = int(self.tgt_combo.currentText())
      if a == b:
        if self.game.num_qubits > 1:
          b = (a + 1) % self.game.num_qubits
          self.tgt_combo.setCurrentIndex(b)
        else:
          self.set_status("Not enough qubits for CX/CZ.")
          return
      qs = [a, b]
      self.set_status(f"{card.operation_name}: control={a}  target={b}")

    # CCX / CCZ
    elif card.operation_name in ('CCX', 'CCZ'):
      a = int(self.ctrl1_combo.currentText())
      b = int(self.ctrl2_combo.currentText())
      c = int(self.tgt_combo.currentText())
      # Ensure all distinct
      if len({a, b, c}) < 3:
        n = self.game.num_qubits
        if n < 3:
          self.set_status("Not enough qubits for CCX/CCZ.")
          return
        if b == a: 
          b = (a + 1) % n
        if c in (a, b): c = (max(a, b) + 1) % n
        self.ctrl2_combo.setCurrentIndex(b)
        self.tgt_combo.setCurrentIndex(c)
      qs = [a, b, c]
      self.set_status(f"{card.operation_name}: controls={a},{b}  target={c}")

    # SWAP
    elif card.operation_name == 'SWAP':
      a = int(self.ctrl1_combo.currentText())
      b = int(self.tgt_combo.currentText())
      if a == b:
        if self.game.num_qubits > 1:
          b = (a + 1) % self.game.num_qubits
          self.tgt_combo.setCurrentIndex(b)
        else:
          self.set_status("Not enough qubits for SWAP.")
          return
      qs = [a, b]
      self.set_status(f"SWAP: qubits {a} ↔ {b}")

    else:
      qs = self.selected_qubits()
      if len(qs) != need:
        self.set_status(f"'{card.operation_name}' needs {need} qubit(s); you selected {len(qs)}.")
        return

    grover_string = None
    if card.operation_name == 'GROVER':
      grover_string, ok = QtWidgets.QInputDialog.getText(
        self, "GROVER", f"Enter a {self.game.num_qubits}-bit string (0/1):")
      if not ok:
        return
      grover_string = str(grover_string).strip()

    try:
      self.game.apply_move(card_played=card, qubit_numbers=qs, grover_string=grover_string)
    except Exception as e:
      self.set_status(f"Error: {e}")
      return

    # Remove the used card from the deck
    deck.pop(row)
    view.remove_row(row)

    # Update UI
    self.set_status(f"Applied {card.operation_name} on {qs}.")
    self.refresh_state_plot()

    # Auto-measure if both players run out of cards
    if self.decks_empty():
      self.lock_after_end()
      self.on_measure()
    else:
      self.next_turn()

  def on_skip(self):
    self.set_status("Turn ended (no move).")
    self.next_turn()

  def on_measure(self):
    # Blank the statevector plot at the very end
    self.state_canvas.ax.clear()
    self.state_canvas.fig.canvas.draw_idle()
    counts = self.game.measure_output_end_of_game()
    plot_counts(counts, ax=self.counts_canvas.ax, fig=self.counts_canvas.fig, normalize=True)

    # Score & announce winner
    p1, p2 = self.game.get_target_bitstring_counts(counts)
    result = "Game results in a tie." if p1 == p2 else ("Player 1 wins!" if p1 > p2 else "Player 2 wins!")
    self.set_status(f"# of times P1 measured a target: {p1} | # of times P2 measured a target: {p2} → {result}")

    # Clear after ~30 seconds
    self.counts_timer.start(30_000)


if __name__ == '__main__':
  # Start Qt app
  app = QtWidgets.QApplication(sys.argv)

  # Set app icon based on light/dark mode
  pal = app.palette()
  is_dark = pal.color(QtGui.QPalette.Window).valueF() < 0.5
  app.setWindowIcon(APP_ICON_DARK if is_dark else APP_ICON_LIGHT)

  w = MainWindow()
  w.show()