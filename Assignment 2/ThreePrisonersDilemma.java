import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ThreePrisonersDilemma {

	/*
	 * This Java program models the three-player Prisoner's Dilemma game.
	 * We use the integer "0" to represent cooperation, and "1" to represent
	 * defection.
	 *
	 * Recall that in the 2-players dilemma, U(DC) > U(CC) > U(DD) > U(CD), where
	 * we give the payoff for the first player in the list. We want the three-player
	 * game to resemble the 2-player game whenever one player's response is fixed,
	 * and we also want symmetry, so U(CCD) = U(CDC) etc. This gives the unique
	 * ordering:
	 *
	 * U(DCC) > U(CCC) > U(DDC) > U(CDC) > U(DDD) > U(CDD)
	 *
	 * The payoffs for player 1 are given by the following matrix:
	 */

	static int[][][] payoff = {
			{ { 6, 3 }, // payoffs when first and second players cooperate
					{ 3, 0 } }, // payoffs when first player coops, second defects
			{ { 8, 5 }, // payoffs when first player defects, second coops
					{ 5, 2 } } }; // payoffs when first and second players defect

	/*
	 * payoff[i][j][k] = payoff to player 1 when player 1 plays i,
	 * player 2 plays j, player 3 plays k.
	 */

	// =========================================================================
	// Accumulate Neo_Sean matchup scores across all runs for the report table.
	// Key: "minOppIdx_maxOppIdx", Value: list of Neo_Sean avg scores per match.
	// =========================================================================
	Map<String, ArrayList<Float>> neoMatchupScores = new HashMap<>();

	// =========================================================================
	// Player definitions
	// =========================================================================

	abstract class Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			throw new RuntimeException("You need to override the selectAction method.");
		}

		final String name() {
			String result = getClass().getName();
			return result.substring(result.indexOf('$') + 1);
		}
	}

	// ---- Neo_Sean_Player --------------------------------------------------- //
	class Neo_Sean_Player extends Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {

			// Component 1: Goodwill initialisation — signal cooperative intent
			if (n == 0)
				return 0;

			// Component 2: Chronic defector detection
			// After 8 rounds of data, lock onto defection if any opponent
			// has defected >80% of the time (handles NastyPlayer quickly).
			if (n >= 8) {
				int def1 = 0, def2 = 0;
				for (int i = 0; i < n; i++) {
					if (oppHistory1[i] == 1)
						def1++;
					if (oppHistory2[i] == 1)
						def2++;
				}
				if ((double) def1 / n > 0.8)
					return 1;
				if ((double) def2 / n > 0.8)
					return 1;
			}

			// Component 3: 3-player Tit-for-Tat
			// Cooperate only when BOTH opponents cooperated last round.
			// (Cooperating against even one defector is individually irrational.)
			boolean opp1CoopLast = (oppHistory1[n - 1] == 0);
			boolean opp2CoopLast = (oppHistory2[n - 1] == 0);
			if (opp1CoopLast && opp2CoopLast)
				return 0;

			// Component 4: Olive branch — escape mutual defection spirals
			// If all three players have been stuck defecting for the last 6 rounds,
			// cooperate to attempt a reset (handles noise from RandomPlayer).
			if (n >= 6) {
				int w = Math.min(n, 6);
				int myDef = 0, d1 = 0, d2 = 0;
				for (int i = n - w; i < n; i++) {
					if (myHistory[i] == 1)
						myDef++;
					if (oppHistory1[i] == 1)
						d1++;
					if (oppHistory2[i] == 1)
						d2++;
				}
				if (myDef == w
						&& d1 >= (int) Math.ceil(0.66 * w)
						&& d2 >= (int) Math.ceil(0.66 * w)) {
					return 0; // extend olive branch
				}
			}

			// Default: defect
			return 1;
		}
	}

	// ---- Baseline strategies ----------------------------------------------- //

	class NicePlayer extends Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			return 0;
		}
	}

	class NastyPlayer extends Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			return 1;
		}
	}

	class RandomPlayer extends Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			return (Math.random() < 0.5) ? 0 : 1;
		}
	}

	class TolerantPlayer extends Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			int coop = 0, defect = 0;
			for (int i = 0; i < n; i++) {
				if (oppHistory1[i] == 0)
					coop++;
				else
					defect++;
				if (oppHistory2[i] == 0)
					coop++;
				else
					defect++;
			}
			return (defect > coop) ? 1 : 0;
		}
	}

	class FreakyPlayer extends Player {
		int action;

		FreakyPlayer() {
			action = (Math.random() < 0.5) ? 0 : 1;
		}

		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			return action;
		}
	}

	class T4TPlayer extends Player {
		int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {
			if (n == 0)
				return 0;
			return (Math.random() < 0.5) ? oppHistory1[n - 1] : oppHistory2[n - 1];
		}
	}

	// =========================================================================
	// Match simulation
	// =========================================================================

	float[] scoresOfMatch(Player A, Player B, Player C, int rounds) {
		int[] hA = new int[0], hB = new int[0], hC = new int[0];
		float sA = 0, sB = 0, sC = 0;
		for (int i = 0; i < rounds; i++) {
			int pA = A.selectAction(i, hA, hB, hC);
			int pB = B.selectAction(i, hB, hC, hA);
			int pC = C.selectAction(i, hC, hA, hB);
			sA += payoff[pA][pB][pC];
			sB += payoff[pB][pC][pA];
			sC += payoff[pC][pA][pB];
			hA = extendIntArray(hA, pA);
			hB = extendIntArray(hB, pB);
			hC = extendIntArray(hC, pC);
		}
		return new float[] { sA / rounds, sB / rounds, sC / rounds };
	}

	int[] extendIntArray(int[] arr, int next) {
		int[] result = new int[arr.length + 1];
		for (int i = 0; i < arr.length; i++)
			result[i] = arr[i];
		result[result.length - 1] = next;
		return result;
	}

	// =========================================================================
	// Player factory
	// =========================================================================

	int numPlayers = 7;
	static final int NEO_IDX = 6; // index of Neo_Sean_Player

	Player makePlayer(int which) {
		switch (which) {
			case 0:
				return new NicePlayer();
			case 1:
				return new NastyPlayer();
			case 2:
				return new RandomPlayer();
			case 3:
				return new TolerantPlayer();
			case 4:
				return new FreakyPlayer();
			case 5:
				return new T4TPlayer();
			case 6:
				return new Neo_Sean_Player();
		}
		throw new RuntimeException("Bad argument passed to makePlayer");
	}

	// =========================================================================
	// Main
	// =========================================================================

	public static void main(String[] args) {
		ThreePrisonersDilemma instance = new ThreePrisonersDilemma();

		final int NUM_RUNS = 5;
		float[][] allRunScores = new float[NUM_RUNS][instance.numPlayers];

		System.out.println("=".repeat(65));
		System.out.println("THREE-PLAYER PRISONER'S DILEMMA  |  Neo_Sean_Player  |  " + NUM_RUNS + " runs");
		System.out.println("=".repeat(65));

		for (int run = 0; run < NUM_RUNS; run++) {
			System.out.println("\n--- Run " + (run + 1) + " ---");
			allRunScores[run] = instance.runTournament();
		}

		// ── Section 1: Averaged overall ranking ────────────────────────────────
		instance.printAverageResults(allRunScores, NUM_RUNS);

		// ── Section 2: Neo_Sean per-matchup breakdown (averaged across runs) ───
		instance.printNeoMatchupTable();
	}

	// =========================================================================
	// Tournament runner (one run)
	// =========================================================================

	float[] runTournament() {
		float[] totalScore = new float[numPlayers];

		for (int i = 0; i < numPlayers; i++)
			for (int j = i; j < numPlayers; j++)
				for (int k = j; k < numPlayers; k++) {

					Player A = makePlayer(i);
					Player B = makePlayer(j);
					Player C = makePlayer(k);
					int rounds = 90 + (int) Math.rint(20 * Math.random());
					float[] r = scoresOfMatch(A, B, C, rounds);

					totalScore[i] += r[0];
					totalScore[j] += r[1];
					totalScore[k] += r[2];

					// ── Log Neo_Sean's per-match score ──────────────────────────
					// When Neo_Sean is player i, j, or k, record its avg score
					// alongside the canonical opponent-pair key.
					if (i == NEO_IDX || j == NEO_IDX || k == NEO_IDX) {
						float neoScore;
						int oA, oB;
						if (i == NEO_IDX) {
							neoScore = r[0];
							oA = j;
							oB = k;
						} else if (j == NEO_IDX) {
							neoScore = r[1];
							oA = i;
							oB = k;
						} else {
							neoScore = r[2];
							oA = i;
							oB = j;
						}

						String key = Math.min(oA, oB) + "_" + Math.max(oA, oB);
						neoMatchupScores
								.computeIfAbsent(key, x -> new ArrayList<>())
								.add(neoScore);
					}
				}

		// Print per-run ranking
		printRunResults(totalScore);
		return totalScore;
	}

	// =========================================================================
	// Print helpers
	// =========================================================================

	void printRunResults(float[] totalScore) {
		int[] order = sortDesc(totalScore);
		System.out.printf("  %-25s %s%n", "Player", "Total Score");
		System.out.println("  " + "-".repeat(38));
		for (int idx : order)
			System.out.printf("  %-25s %.2f%n", makePlayer(idx).name(), totalScore[idx]);
	}

	void printAverageResults(float[][] allRunScores, int numRuns) {
		float[] avg = new float[numPlayers];
		for (int i = 0; i < numPlayers; i++) {
			for (int r = 0; r < numRuns; r++)
				avg[i] += allRunScores[r][i];
			avg[i] /= numRuns;
		}
		int[] order = sortDesc(avg);

		System.out.println("\n" + "=".repeat(65));
		System.out.println("SECTION 1 — AVERAGE OVERALL RANKING  (" + numRuns + " runs)");
		System.out.println("=".repeat(65));
		System.out.printf("  %-6s %-25s %12s%n", "Rank", "Player", "Avg Total Score");
		System.out.println("  " + "-".repeat(45));
		for (int rank = 0; rank < numPlayers; rank++) {
			int idx = order[rank];
			System.out.printf("  %-6d %-25s %12.2f%n",
					rank + 1, makePlayer(idx).name(), avg[idx]);
		}
	}

	void printNeoMatchupTable() {
		System.out.println("\n" + "=".repeat(65));
		System.out.println("SECTION 2 — NEO_SEAN_PLAYER MATCHUP SCORES  (avg over all runs)");
		System.out.println("=".repeat(65));
		System.out.printf("  %-22s %-22s %14s%n", "Opponent 1", "Opponent 2", "Neo Avg Score");
		System.out.println("  " + "-".repeat(60));

		// Print all opponent pairs in canonical order
		for (int a = 0; a < numPlayers; a++) {
			for (int b = a; b < numPlayers; b++) {
				String key = a + "_" + b;
				ArrayList<Float> scores = neoMatchupScores.get(key);
				if (scores == null)
					continue;
				double avg = scores.stream().mapToDouble(f -> f).average().orElse(0.0);
				System.out.printf("  %-22s %-22s %14.2f%n",
						makePlayer(a).name(), makePlayer(b).name(), avg);
			}
		}

		System.out.println("\n  Legend:");
		System.out.println("  6.00  = full mutual cooperation sustained");
		System.out.println("  ~5.xx = one defector detected; Neo defects back, limiting loss");
		System.out.println("  3-4   = mixed/noisy environment, olive branch partially recovers");
		System.out.println("  ~2.xx = all-defect equilibrium (chronic defectors present)");
	}

	/** Returns indices sorted in descending order of scores. */
	int[] sortDesc(float[] scores) {
		int[] order = new int[numPlayers];
		for (int i = 0; i < numPlayers; i++) {
			int j = i - 1;
			for (; j >= 0; j--) {
				if (scores[i] > scores[order[j]])
					order[j + 1] = order[j];
				else
					break;
			}
			order[j + 1] = i;
		}
		return order;
	}

} // end of class ThreePrisonersDilemma