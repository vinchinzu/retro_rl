# In-game hour stops at 18; D2 work does not end at night

`UpdateTime` (`HM-Decomp` `bank_82.asm`) increments hour until it equals 18, then
skips the increment. Minutes still wrap; the day does not roll. D2 farm clear
therefore runs until debris is gone, including after 18:00. A “six hour” 60fps
tape length is a guess, not an acceptance timeout. Shipping still has to land
before 17:00; after that the bot may smash until the farm is clear.
