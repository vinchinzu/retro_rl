package com.smb.editor.cli

import com.smb.editor.rom.*
import com.smb.editor.data.*
import java.awt.Color
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File

object ImageRenderer {

    fun renderToImage(rom: NesRom, entry: AreaEntry, overlays: Set<TileOverlay> = emptySet()): BufferedImage {
        val level = LevelDecoder.decodeLevel(rom, entry)
        val result = LevelRenderer.renderLevel(rom, level)

        val image = BufferedImage(result.width, result.height, BufferedImage.TYPE_INT_ARGB)
        image.setRGB(0, 0, result.width, result.height, result.pixels, 0, result.width)

        if (overlays.isNotEmpty()) {
            val g = image.createGraphics()
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g.font = Font("SansSerif", Font.PLAIN, 9)

            val items = OverlayGenerator.generate(level, overlays)
            for (item in items) {
                val color = Color(
                    (item.color shr 16) and 0xFF,
                    (item.color shr 8) and 0xFF,
                    item.color and 0xFF,
                    (item.color ushr 24) and 0xFF
                )

                when (item.type) {
                    OverlayType.OBJECT_BOUNDS -> {
                        g.color = Color(0, 255, 0, 128)
                        g.drawRect(item.x, item.y, item.width - 1, item.height - 1)
                        g.color = Color(0, 255, 0, 200)
                        g.drawString(item.label, item.x, item.y - 2)
                    }
                    OverlayType.ENEMY_MARKER -> {
                        g.color = Color(255, 0, 0, 128)
                        g.fillOval(item.x + 2, item.y + 2, item.width - 4, item.height - 4)
                        g.color = Color(255, 0, 0, 200)
                        g.drawString(item.label, item.x, item.y - 2)
                    }
                    OverlayType.GRID_LINE -> {
                        g.color = Color(255, 255, 255, 40)
                        g.drawLine(item.x, item.y, item.x + item.width, item.y + item.height)
                    }
                    else -> {
                        g.color = color
                        g.drawRect(item.x, item.y, item.width - 1, item.height - 1)
                    }
                }
            }

            // Draw grid lines if requested
            if (TileOverlay.GRID in overlays) {
                g.color = Color(255, 255, 255, 30)
                // Vertical lines every 16px
                for (x in 0 until result.width step 16) {
                    g.drawLine(x, 0, x, result.height)
                }
                // Horizontal lines every 16px
                for (y in 0 until result.height step 16) {
                    g.drawLine(0, y, result.width, y)
                }
                // Page boundaries every 256px (16 metatiles)
                g.color = Color(255, 255, 0, 60)
                for (x in 0 until result.width step 256) {
                    g.drawLine(x, 0, x, result.height)
                }
            }

            g.dispose()
        }

        return image
    }

    fun saveImage(image: BufferedImage, path: String) {
        ImageIO.write(image, "PNG", File(path))
    }
}
