## QARDS
A quantum card game

### Rules
1. There are 3-5 qubits, starting in a random (possibly entangled) state.
2. Each player has a number of target bitstrings whose length is the number of qubits in the game.
   If one of these is measured, that player earns a point.
3. Each player has a number of cards that apply quantum operations.
4. Players alternate turns playing cards. When both decks are empty, the qubits are measured, and the player whose bitstrings appear most often wins.

Card Reference:
    - X, Y, Z, H, CX, CCX, CZ, CCZ, S, T, S_dag, T_dag, I, SWAP:
        Apply the corresponding quantum gate on chosen qubit(s).

    - DIFFUSION: Grover diffusion operator.

    - RESET: Resets a qubit to |0⟩ (randomly flips to |1⟩ half the time).

    - GROVER: Runs a Grover iteration amplifying a bitstring chosen by the player.
    
    - REVERSE: Undoes the previous move (restores the prior circuit).

Note: Bitstrings follow little-endian order (Qubit 0 is rightmost).

### How to run
After pulling the repository

First, create a virtual environment:

```
python3 -m venv .venv
```

Activate the virtual environment (Linux example):

```
source .venv/bin/activate
```

Install the required libraries:

```
pip install -r requirements.txt
```

Play the game:

```
python3 QARDS_GUI.py
```