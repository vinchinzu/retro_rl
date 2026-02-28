package com.smb.editor.cli

import com.smb.editor.rom.*
import com.smb.editor.data.*
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        printUsage()
        exitProcess(1)
    }

    val iter = args.iterator()
    var romPath: String? = null
    var command: String? = null
    val commandArgs = mutableListOf<String>()

    while (iter.hasNext()) {
        val arg = iter.next()
        when {
            arg == "--rom" && iter.hasNext() -> romPath = iter.next()
            arg == "--help" || arg == "-h" -> { printUsage(); return }
            command == null -> command = arg
            else -> commandArgs.add(arg)
        }
    }

    if (romPath == null) {
        System.err.println("Error: --rom <path> is required")
        exitProcess(1)
    }

    val romData = try {
        java.io.File(romPath).readBytes()
    } catch (e: Exception) {
        System.err.println("Error reading ROM: ${e.message}")
        exitProcess(1)
    }

    val rom = try {
        NesRom.fromBytes(romData)
    } catch (e: Exception) {
        System.err.println("Error parsing ROM: ${e.message}")
        exitProcess(1)
    }

    println("Loaded ROM: ${romPath} (PRG: ${rom.prgBanks * 16}KB, CHR: ${rom.chrBanks * 8}KB)")

    when (command) {
        "levels" -> cmdLevels(rom)
        "level" -> cmdLevel(rom, commandArgs)
        "render" -> cmdRender(rom, commandArgs)
        "export" -> cmdExport(rom, commandArgs)
        null -> { System.err.println("Error: no command specified"); printUsage(); exitProcess(1) }
        else -> { System.err.println("Error: unknown command '$command'"); printUsage(); exitProcess(1) }
    }
}

private fun printUsage() {
    println("""
        |Super Mario Bros ROM Editor CLI
        |
        |Usage: smb-editor --rom <path.nes> <command> [options]
        |
        |Commands:
        |  levels                          List all levels
        |  level <world-level>             Show level details (e.g. "1-1")
        |  render <world-level> [options]  Render level to PNG
        |    -o <output.png>               Output file (default: <level>.png)
        |    --overlay <types>             Comma-separated: enemies,objects,grid,coins,powerups,pipes,warps
        |  export [-o <dir>]               Export all levels to JSON
    """.trimMargin())
}

private fun cmdLevels(rom: NesRom) {
    println()
    println("%-8s %-25s %-12s %s".format("Level", "Name", "Area Type", "Area #"))
    println("-".repeat(60))
    for (entry in LevelRegistry.LEVELS) {
        val typeName = LevelRegistry.AREA_TYPE_NAMES[entry.areaType] ?: "Unknown"
        println("%-8s %-25s %-12s 0x%02X".format(
            entry.worldLevel, entry.name, typeName, entry.areaNumber
        ))
    }
    println()
    println("Total: ${LevelRegistry.LEVELS.size} levels")
}

private fun cmdLevel(rom: NesRom, args: List<String>) {
    if (args.isEmpty()) {
        System.err.println("Error: level command requires a world-level argument (e.g. '1-1')")
        exitProcess(1)
    }

    val worldLevel = args[0]
    val entry = LevelRegistry.findByWorldLevel(worldLevel)
    if (entry == null) {
        System.err.println("Error: unknown level '$worldLevel'")
        exitProcess(1)
    }

    val level = LevelDecoder.decodeLevel(rom, entry)

    println()
    println("Level: ${level.worldLevel} - ${level.name}")
    println("Area Type: ${LevelRegistry.AREA_TYPE_NAMES[level.header.areaType]} (${level.header.areaType})")
    println("Area Style: ${level.header.platformType}")
    println("Terrain Control: ${level.header.terrainControl}")
    println("Cloud Override: ${level.header.cloudTypeOverride}")
    println("Background Scenery: ${level.header.bgScenery}")
    println("Background Color: ${level.header.bgColor}")
    println("Time Settings: ${level.header.timeSettings}")
    println("FG Scenery: ${level.header.fgScenery}")
    println()
    println("Objects: ${level.objects.size}")
    for ((i, obj) in level.objects.withIndex()) {
        val name = ObjectCatalog.getObjectName(obj)
        println("  [$i] Page ${obj.page}, Row ${obj.row}, Col ${obj.col}: $name (type=0x${obj.type.toString(16).uppercase()})")
    }
    println()
    println("Enemies: ${level.enemies.size}")
    for ((i, enemy) in level.enemies.withIndex()) {
        val name = EnemyCatalog.getEnemyName(enemy.type)
        println("  [$i] Page ${enemy.page}, Row ${enemy.row}, Col ${enemy.col}: $name (type=0x${enemy.type.toString(16).uppercase()})")
    }
}

private fun cmdRender(rom: NesRom, args: List<String>) {
    if (args.isEmpty()) {
        System.err.println("Error: render command requires a world-level argument")
        exitProcess(1)
    }

    val iter = args.iterator()
    val worldLevel = iter.next()
    var outputPath: String? = null
    var overlayStr: String? = null

    while (iter.hasNext()) {
        val arg = iter.next()
        when {
            arg == "-o" && iter.hasNext() -> outputPath = iter.next()
            arg == "--overlay" && iter.hasNext() -> overlayStr = iter.next()
        }
    }

    val entry = LevelRegistry.findByWorldLevel(worldLevel)
    if (entry == null) {
        System.err.println("Error: unknown level '$worldLevel'")
        exitProcess(1)
    }

    val overlays = parseOverlays(overlayStr)
    val output = outputPath ?: "${worldLevel.replace("-", "_")}.png"

    val image = ImageRenderer.renderToImage(rom, entry, overlays)
    ImageRenderer.saveImage(image, output)

    println("Rendered ${worldLevel} to $output (${image.width}x${image.height})")
}

private fun cmdExport(rom: NesRom, args: List<String>) {
    val iter = args.iterator()
    var outputDir = "."

    while (iter.hasNext()) {
        val arg = iter.next()
        when {
            arg == "-o" && iter.hasNext() -> outputDir = iter.next()
        }
    }

    val dir = java.io.File(outputDir)
    if (!dir.exists()) dir.mkdirs()

    val exporter = LevelExporter()
    for (entry in LevelRegistry.LEVELS) {
        val export = exporter.exportLevel(rom, entry)
        val json = export.toJson()
        val file = java.io.File(dir, "${entry.worldLevel.replace("-", "_")}.json")
        file.writeText(json)
        println("Exported ${entry.worldLevel} to ${file.path}")
    }

    println()
    println("Exported ${LevelRegistry.LEVELS.size} levels to $outputDir")
}

private fun parseOverlays(str: String?): Set<TileOverlay> {
    if (str == null) return emptySet()
    return str.split(",").mapNotNull { name ->
        when (name.trim().lowercase()) {
            "enemies" -> TileOverlay.ENEMIES
            "objects" -> TileOverlay.OBJECTS
            "grid" -> TileOverlay.GRID
            "coins" -> TileOverlay.COINS
            "powerups" -> TileOverlay.POWERUPS
            "pipes" -> TileOverlay.PIPES
            "warps", "warp_zones" -> TileOverlay.WARP_ZONES
            "all" -> null  // handled separately
            else -> { System.err.println("Warning: unknown overlay '$name'"); null }
        }
    }.toSet().let { set ->
        if (str.contains("all", ignoreCase = true)) TileOverlay.values().toSet()
        else set
    }
}
