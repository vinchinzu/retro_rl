package com.smb.editor.data

import kotlinx.serialization.Serializable

@Serializable
data class LevelHeader(
    val fgScenery: Int,         // header byte 0 low bits (0-3): foreground scenery selector
    val bgScenery: Int,         // header byte 1 bits 5-4: background scenery selector
    val platformType: Int,      // SMB area style (stored for compatibility with existing UI name)
    val terrainControl: Int = 0, // header byte 1 low nybble: terrain pattern selector
    val cloudTypeOverride: Boolean = false, // header byte 1 bits 7-6 == 3
    val timeSettings: Int,      // header byte 0 bits 7-6: timer settings
    val bgColor: Int,           // header byte 0 low bits when >= 4: background color control
    val areaType: Int           // byte 1, bits 2-0: area type (0=water,1=overground,2=underground,3=castle)
)

@Serializable
data class LevelObject(
    val row: Int,       // 0-13
    val col: Int,       // 0-15 within page
    val page: Int,      // page number
    val type: Int,      // object type code
    val param: Int = 0  // extra parameter (length for expandable objects)
)

@Serializable
data class Enemy(
    val row: Int,
    val col: Int,
    val page: Int,
    val type: Int
)

@Serializable
data class LevelData(
    val header: LevelHeader,
    val objects: List<LevelObject>,
    val enemies: List<Enemy>,
    val worldLevel: String,     // "1-1", "8-4" etc.
    val name: String,           // display name
    val areaIndex: Int          // index into area data tables
)
