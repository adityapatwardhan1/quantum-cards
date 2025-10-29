# THIS IS A CLI VERSION IT IS BETTER

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, random_unitary
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt

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

statevector_plot_fig = None
statevector_plot_axes = None

counts_plot_fig = None
counts_plot_axes = None

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

def get_qubit_number(num_qubits: int):
    """Reads & validates input for what qubit the user wants the operation
    to perform on"""
    qubit_num = ""
    while True:
      try:
        print(f"Enter a number between 0 and {num_qubits - 1} " +
          "inclusive")
        the_input = input()
        qubit_num = int(the_input)
        if qubit_num < 0 or qubit_num >= num_qubits:
          continue
        else:
          break
      except ValueError:
        print("Not a valid integer input")
    return qubit_num

def get_n_qubit_numbers(num_qubits: int, n: int):
  """Reads input for what qubits the user wants operation to perform on
  The qubits involved in the operation must be distinct"""
  qubit_numbers = []
  for i in range(n):
    qubit_num = get_qubit_number(num_qubits)
    while qubit_num in qubit_numbers:
      print("You have already selected that qubit, try again")
      qubit_num = get_qubit_number(num_qubits)
    qubit_numbers.append(qubit_num)
  return qubit_numbers

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

def print_cards(current_player_deck: list):
    print("[", end="")
    for i in range(len(current_player_deck) - 1):
      item = current_player_deck[i]
      print(f"{i}:{item}, ", end="")
    print(f"{len(current_player_deck) - 1}:{current_player_deck[-1]}]")

def get_deck_operation(card_deck: list):
  """Removes and returns a card from the deck based on user input"""
  card_num = -1
  # Input validation loop
  while card_num < 0 or card_num >= len(card_deck):
    try:
        print(f"Select a card from the deck by 0-based index: ", end="")
        print_cards(card_deck)
        the_input = input()
        card_num = int(the_input)
        if card_num < 0 or card_num >= len(card_deck):
          print("Not a valid index in this deck")
          continue
        else:
          break
    except ValueError:
      print("Not a valid integer input")
  return card_deck.pop(card_num)

def plot_statevector_data(statevector_dict, thresh_to_plot=0.02):
  """
  Plots a statevector as a bar graph
  :param statevector: Vector representing a quantum state
  :type statevector: list
  :param thresh_to_plot: Minimum magnitude for a statevector entry to be plotted.
    This reduces the number of entries to plot which is better visually.
  :type thresh_to_plot: float
  """
  global statevector_plot_axes, statevector_plot_fig
  if statevector_plot_fig is None or statevector_plot_axes is None:
        plt.ion()
        statevector_plot_fig, statevector_plot_axes = plt.subplots(figsize=(10, 5))
        statevector_plot_fig.canvas.manager.set_window_title("Statevector Probabilities")
  
  statevector_plot_axes.clear()

  # Get filtered & sorted keys/values
  keys = sorted(statevector_dict.keys())
  values = [statevector_dict[key] for key in keys]
  
  # Set ticks first, then labels
  import numpy as np
  x = np.arange(len(keys)) # Fixed tick positions
  statevector_plot_axes.bar(keys, values, width=0.5, color='blue')
  statevector_plot_axes.set_xticks(x)
  statevector_plot_axes.set_xticklabels(keys, rotation=45)
  
  # Axis labels & limits
  statevector_plot_axes.set_ylabel('|Probability Amplitude|^2')
  statevector_plot_axes.set_xlabel('Measurement Outcome')
  statevector_plot_axes.set_ylim([0, 1])

  statevector_plot_fig.tight_layout()
  statevector_plot_fig.canvas.draw()
  statevector_plot_fig.canvas.flush_events()
  
  plt.pause(0.1)

def plot_counts(counts:dict, thresh_to_plot:float=0.00, normalize:bool=False):
  """
  Plots the measurement oucome as a bar graph
  :param counts: Vector representing counts after running a quantum circuit
  :type counts: dict
  :param thresh_to_plot: Minimum fraction of the times that a measurement
    outcome must occur among all the measurement outcomes to be plotted.
    This reduces the number of entries to plot which is better visually.
  :type thresh_to_plot: float
  :param normalize: Whether to normalize the counts to fractions of total counts
  :type normalize: bool
  """
  import numpy as np
  global counts_plot_axes, counts_plot_fig
  
  # Compute total counts
  total = sum(counts.values()) if counts else 0
  if (total == 0):
      print("No counts to plot")
      return
  
  # (key, value) pairs; normalize if requested
  items = [(k, (v/total) if normalize else v) for k, v in counts.items()]
  if (normalize):
      items = [(k, v) for k, v in items if v >= thresh_to_plot]
  
  # Sort by integer value of the bitstring
  items.sort(key=lambda kv: int(kv[0], 2))
  
  # Extract keys & vals
  keys = [k for k, _ in items]
  vals = [v for _, v in items]
  x = np.arange(len(keys))

  # Create figure/axes
  plt.ion()
  counts_plot_fig, counts_plot_axes = plt.subplots(figsize=(10, 5))
  counts_plot_fig.canvas.manager.set_window_title("Measurement Results")

  # Clear axes
  ax = counts_plot_axes
  ax.clear()
  
  # Red bars
  bars = ax.bar(x, vals, width=0.5, color='red')
  
  # Axis labels
  ax.set_ylabel('Counts')
  ax.set_xlabel('Measurement Outcome')
  
  # Tick labels
  ax.set_xticks(x)
  ax.set_xticklabels(keys, rotation=45, ha='right')

  # Axis labels & limits with headroom for text labels
  if normalize:
    ax.set_ylabel('Fraction of shots')
    # Headroom above tallest bar for label text
    ymax = max(vals) if vals else 1
    ax.set_ylim(0, max(1.0, ymax * 1.15))
  else:
    ax.set_ylabel('Counts')
    ymax = max(vals) if vals else 1
    ax.set_ylim(0, ymax * 1.15)

  # Value labels above bars
    for rect, v in zip(bars, vals):
        height = rect.get_height()
        if normalize and as_percent:
            label = f"{v:.1%}"
        elif normalize:
            label = f"{v:.3f}"
        else:
            label = f"{int(v)}"
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 px offset above the bar
            textcoords="offset points",
            ha="center", va="bottom"
        )

    counts_plot_fig.tight_layout()
    counts_plot_fig.canvas.draw()
    counts_plot_fig.canvas.flush_events()
    plt.pause(0.1)


def clear_terminal():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

def verify_bitstring_of_length_n(bitstring: str, n: int):
  if len(bitstring) != n:
    return False
  for i in range(len(bitstring)):
    if not bitstring[i] in ['0', '1']:
      return False
  return True

def get_grover_input(num_qubits: int):
  cur = ""
  while not verify_bitstring_of_length_n(cur, num_qubits):
    cur = input(f"Enter a bitstring whose length is {num_qubits} that you want to amplify:")
  return cur

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
    # Default number of target bitstrings if not specified
    if num_bitstrings_per_player == 0:
      num_bitstrings_per_player = int(2**(num_qubits - 2))

    # Initialize the quantum circuit using a random unitary
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

    # Generate target bitstrings
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

  def print_game_rules(self):
    print("Game Rules:")
    print("1. There are", self.num_qubits, "qubits, forming a random (possibly entangled) state.")
    print("2. Each player has", self.num_target_bitstrings, "target bitstrings of length",
          self.num_qubits, "which, if measured, earn them points.")
    print("   The player with the most points after measuring the qubits wins.")
    print("3. Each player has", self.num_cards_per_player, "cards that correspond to quantum gates.")
    print("   On your turn, select a card from your deck to play.")
    print("4. Once all cards have been played, the qubits are measured and the winner is determined.")
    print("Note: Bitstrings in the game are in little-endian convention, so Qubit 0 is the rightmost qubit.")
    time.sleep(5)
    clear_terminal()

  def before_game_loop(self):
    """Prints information for the players before the game starts"""
    print("Welcome to QARDS!")
    self.print_game_rules()
    print("Cards for player 1:", self.deck_player_one)
    print("Cards for player 2:", self.deck_player_two)
    print("Bitstrings for player 1:", self.bitstrings_player_one)
    print("Bitstrings for player 2:", self.bitstrings_player_two)
    print("Flipping coin to determine who goes first...")
    time.sleep(2)
    print(f"The first turn goes to: Player {self.turn}")
    time.sleep(2)
    clear_terminal()
    statevector_data = self.get_statevector_data()
    plot_statevector_data(statevector_data)

  def apply_grover_oracle_with_planted_sol(self, bitstring: str):
    """
    Applies a Grover oracle with the planted solution 
    corresponding to bitstring
    :param bitstring: The bitstring to perform an iteration of Grover for
    """
    # Apply X gates when you want that certain bit to be 0
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

  def apply_move(self, card_played: Card):
    self.list_of_previous_circuits.append(self.qc)
    self.qc = self.qc.copy()
    
    qubit_numbers = []
    qubit_numbers = get_n_qubit_numbers(self.num_qubits,
                                        MOVE_TO_NUM_QUBITS[card_played.operation_name])

    match card_played.operation_name:
      case 'X':
        self.qc.x(self.q[qubit_numbers[0]])
      case 'Y':
        self.qc.y(self.q[qubit_numbers[0]])
      case 'Z':
        self.qc.z(self.q[qubit_numbers[0]])
      case 'H':
        self.qc.h(self.q[qubit_numbers[0]])
      case 'CX':
        self.qc.cx(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]])
      case 'CZ':
        self.qc.cz(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]])
      case 'CCX':
        self.qc.ccx(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]], self.q[qubit_numbers[2]])
      case 'CCZ':
        self.qc.ccz(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]], self.q[qubit_numbers[2]])
      case 'S':
        self.qc.s(self.q[qubit_numbers[0]])
      case 'S_dag':
        self.qc.sdg(self.q[qubit_numbers[0]])
      case 'T':
        self.qc.t(self.q[qubit_numbers[0]])
      case 'T_dag':
        self.qc.tdg(self.q[qubit_numbers[0]])
      case 'SWAP':
        self.qc.swap(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]])
      case 'DIFFUSION':
        self.qc.h(self.q)
        self.qc.x(self.q)
        self.qc.mcp(np.pi,self.q[1:],self.q[0])
        self.qc.x(self.q)
        self.qc.h(self.q)
      case 'RESET':
        self.qc.reset(self.q[qubit_numbers[0]])
        if random.random() < 0.5:
          self.qc.x(self.q[qubit_numbers[0]])
      case 'GROVER':
        bitstring = get_grover_input(self.num_qubits)
        self.apply_grover_oracle_with_planted_sol(bitstring)
        self.diffusion_operator()
      case 'REVERSE':
        if len(self.list_of_previous_circuits) > 1:
          self.qc = self.list_of_previous_circuits[-2].copy()
        


  def measure_output_end_of_game(self):
    """Measures output at the end of the game"""
    for i in range(self.num_qubits):
      self.qc.measure(self.q[i], self.c[i])
    backend = Aer.get_backend('qasm_simulator')
    NUM_SHOTS = 1024
    result = backend.run(self.qc, shots=NUM_SHOTS).result()
    counts = result.get_counts()
    return counts

  def get_statevector_data(self):
    """Gets the statevector data after all the moves that have been applied
    thus far"""
    v_init = Statevector.from_label('0' * self.num_qubits)
    v_evolved = v_init.evolve(self.qc)
    return v_evolved.probabilities_dict()

  def print_statevector_data(self, statevector_data):
    """
    Prints entries of a statevector
    :param statevector_data: Vector representing a quantum state
    :type statevector_data: list
    """
    print(statevector_data)
    # for idx, data in enumerate(statevector_data):
    #   print(str(bin(idx)[2:]).zfill(self.num_qubits), data)

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

  def get_winner(self, counts):
    """
    Determines winner of the game based on measurement counts of the circuit.
    The player who has one of their target bitstrings occur more times wins.
    :param counts: Number of occurrences of each measured outcome
    :type counts: dict
    :returns: 1 if player 1 wins, 2 if player 2 wins, 3 if tie.
    """
    count_player_one, count_player_two = self.get_target_bitstring_counts(counts)
    if count_player_one > count_player_two:
      return 1
    elif count_player_one < count_player_two:
      return 2
    else:
      return 3

  def handle_player_turn(self):
    print(f"Turn: Player {self.turn}")
    current_player_deck = (self.deck_player_one if self.turn == 1 else self.deck_player_two)
    current_player_bistrings = (self.bitstrings_player_one if self.turn == 1
                                else self.bitstrings_player_two)
    print("Your bitstrings:", current_player_bistrings)
    card_played = get_deck_operation(current_player_deck)
    # Apply unitary corresponding to card
    self.apply_move(card_played)
    statevector_data = self.get_statevector_data()
    plot_statevector_data(statevector_data)
    # self.print_statevector_data(statevector_data)
    time.sleep(3)
    clear_terminal()
    return card_played

  def game_loop(self):
    loop_counter = self.num_cards_per_player
    while loop_counter > 0:
      print("Number of turns left:", loop_counter)
      loop_counter -= 1
      self.last_card = self.handle_player_turn()
      self.turn = 2 if self.turn == 1 else 1
      self.last_card = self.handle_player_turn()
      self.turn = 2 if self.turn == 1 else 1

  def end_of_game(self):
    """Measures circuit and prints who won or if there was a tie"""
    print("Game has ended, measuring the circuit...")
    counts = self.measure_output_end_of_game()

    # Score & announce winner
    p1, p2 = self.get_target_bitstring_counts(counts)
    print("# of times player 1 measured one of their target bitstrings:", p1)
    print("# of times player 2 measured one of their target bitstrings:", p2)
    if p1 > p2:
        print("Player 1 wins!")
    elif p1 < p2:
        print("Player 2 wins!")
    else:
        print("Game results in a tie.")
        
    # Close statevector plot
    global statevector_plot_fig
    plt.close(statevector_plot_fig)
    statevector_plot_fig = None
    
    # Plot measurement results
    plot_counts(counts)
    
    # # Keep only the counts window up for ~ 5 minutes, allow dragging/resizing, then close cleanly.
    global counts_plot_fig
    if counts_plot_fig is not None and plt.fignum_exists(counts_plot_fig.number):
      timer = counts_plot_fig.canvas.new_timer(interval=10000)  # 10 seconds
      timer.single_shot = True
      timer.add_callback(lambda: (plt.close(counts_plot_fig)))
      timer.start()
    
    # Blocks while the GUI stays responsive
    print("Showing measurement results")
    plt.ioff()
    plt.show()

  def play_game(self):
    self.before_game_loop()
    self.game_loop()
    self.end_of_game()

if __name__ == '__main__':
    num_qubits = 0
    while True:
      try:
        num_qubits = int(input("Enter number of qubits you want to play the game using (3 - 7): "))
        if not (3 <= num_qubits <= 7):
          print("Enter a positive integer between 3 and 7, inclusive.")
        else:
          break
      except ValueError:
        print("Enter a positive integer between 3 and 7, inclusive.")
    
    num_cards = 0
    while True:
      try:
        num_cards = int(input("Enter number of cards you want to play the game using (1 - 20): "))
        if num_cards < 1:
          print("You must play with at least one card.")
        elif num_cards > 20:
          print("Pick a lower number of cards (max 20).")
        else:
          break
      except ValueError:
        print("Enter a positive integerbetween 1 and 20, inclusive.")

    num_bitstrings_per_player = 0
    while True:
      try:
        num_bitstrings_per_player = int(input(f"Enter number of bitstrings per player (Enter 0 for default: 2^{num_qubits - 2}): "))
        if num_bitstrings_per_player > int(2**(num_qubits - 1)):
          print("Pick a lower number of bitstrings per player. Each player can have at most half the bitstrings.")
        break
      except ValueError:
        print("Enter a positive integer.")
    
    clear_terminal()
    # Play game
    game = Game(num_qubits, num_cards, num_bitstrings_per_player=num_bitstrings_per_player)
    game.play_game()