class Neo_Sean_Player extends Player {
    int selectAction(int n, int[] myHistory, int[] oppHistory1, int[] oppHistory2) {

        // Round 0 signal goodwill
        if (n == 0)
            return 0;

        // Detect chronic defectors like nasty players
        // After enough data, if someone defects >80% of the time, they're hostile.
        // No point cooperating
        if (n >= 8) {
            int totalDef1 = 0;
            int totalDef2 = 0;

            for (int i = 0; i < n; i++) {
                if (oppHistory1[i] == 1)
                    totalDef1++;
                if (oppHistory2[i] == 1)
                    totalDef2++;
            }

            if ((double) totalDef1 / n > 0.8)
                return 1;
            if ((double) totalDef2 / n > 0.8)
                return 1;
        }

        // 3-player T4T
        // in 3 player, even ONE defector makes cooperation costly for us
        // cooperator only if both cooperatedd last round
        boolean opp1CoopLast = (oppHistory1[n - 1] == 0);
        boolean opp2CoopLast = (oppHistory2[n - 1] == 0);

        if (opp1CoopLast && opp2CoopLast)
            return 0; // cooperate if both cooperated last round

        if (n >= 6) {
            int windowSize = Math.min(n, 6);
            int myRecentDef = 0, recentDef1 = 0, recentDef2 = 0;

            for (int i = n - windowSize; i < n; i++) {
                if (myHistory[i] == 1)
                    myRecentDef++;
                if (oppHistory1[i] == 1)
                    recentDef1++;
                if (oppHistory2[i] == 1)
                    recentDef2++;
            }

            // All 3 have been stuck defecting, try cooperating to break the cycle
            // We defected 5/6 rounds, opponents defected at least 4/6 rounds
            if (myRecentDef == windowSize && recentDef1 >= Math.ceil(0.66 * windowSize)
                    && recentDef2 >= Math.ceil(0.66 * windowSize)) {
                return 0; // cooperate to break the cycle
            }
        }

        // Default is to defect
        return 1;
    }
}