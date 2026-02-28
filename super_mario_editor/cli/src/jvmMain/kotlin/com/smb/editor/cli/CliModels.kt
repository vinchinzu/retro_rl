package com.smb.editor.cli

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

private val prettyJson = Json { prettyPrint = true }

@Serializable
data class LevelExportModel(
    val worldLevel: String,
    val name: String,
    val areaType: Int,
    val areaTypeName: String,
    val header: HeaderExport,
    val objects: List<ObjectExport>,
    val enemies: List<EnemyExport>,
    val gridWidth: Int,
    val gridHeight: Int
) {
    fun toJson(): String = prettyJson.encodeToString(this)
}

@Serializable
data class HeaderExport(
    val areaType: Int,
    val fgScenery: Int,
    val bgScenery: Int,
    val platformType: Int,
    val terrainControl: Int,
    val cloudTypeOverride: Boolean,
    val bgColor: Int,
    val timeSettings: Int
)

@Serializable
data class ObjectExport(
    val index: Int,
    val row: Int,
    val col: Int,
    val page: Int,
    val type: Int,
    val typeHex: String,
    val typeName: String,
    val param: Int
)

@Serializable
data class EnemyExport(
    val index: Int,
    val row: Int,
    val col: Int,
    val page: Int,
    val type: Int,
    val typeHex: String,
    val typeName: String
)

@Serializable
data class LevelSummary(
    val worldLevel: String,
    val name: String,
    val areaType: String,
    val objectCount: Int,
    val enemyCount: Int
)
