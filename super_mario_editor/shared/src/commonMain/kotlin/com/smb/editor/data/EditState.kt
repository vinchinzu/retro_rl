package com.smb.editor.data

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
sealed class LevelEdit {
    @Serializable
    data class AddObject(val obj: LevelObject, val index: Int) : LevelEdit()

    @Serializable
    data class RemoveObject(val obj: LevelObject, val index: Int) : LevelEdit()

    @Serializable
    data class MoveObject(
        val index: Int,
        val oldRow: Int, val oldCol: Int, val oldPage: Int,
        val newRow: Int, val newCol: Int, val newPage: Int
    ) : LevelEdit()

    @Serializable
    data class ChangeObjectType(
        val index: Int,
        val oldType: Int, val oldParam: Int,
        val newType: Int, val newParam: Int
    ) : LevelEdit()

    @Serializable
    data class AddEnemy(val enemy: Enemy, val index: Int) : LevelEdit()

    @Serializable
    data class RemoveEnemy(val enemy: Enemy, val index: Int) : LevelEdit()

    @Serializable
    data class MoveEnemy(
        val index: Int,
        val oldRow: Int, val oldCol: Int, val oldPage: Int,
        val newRow: Int, val newCol: Int, val newPage: Int
    ) : LevelEdit()

    @Serializable
    data class ChangeEnemyType(
        val index: Int, val oldType: Int, val newType: Int
    ) : LevelEdit()
}

@Serializable
data class SmbEditProject(
    val romPath: String,
    val levelEdits: Map<String, List<LevelEdit>> = emptyMap()
) {
    companion object {
        private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }

        fun fromJson(text: String): SmbEditProject = json.decodeFromString(text)
    }

    fun toJson(): String = Companion.json.encodeToString(serializer(), this)
}
